"""CPU checks of the actual adapter hook without loading model weights."""
import unittest
try:
    import torch
    from aw.model import OmniAdapter
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class InterventionTests(unittest.TestCase):
    def model(self):
        adapter = OmniAdapter.__new__(OmniAdapter)
        adapter.device = torch.device('cpu')
        adapter.layers = [torch.nn.Identity() for _ in range(4)]
        return adapter

    def test_only_last_token_is_edited_and_scale_matches(self):
        adapter = self.model()
        h = torch.tensor([[[3., 4.], [3., 4.]]])
        v = torch.tensor([[1., 0.]] * 4)
        with adapter.intervention(v, [1], .12) as receipt:
            y = adapter.layers[0](h)
        torch.testing.assert_close(y[:, 0], h[:, 0])
        torch.testing.assert_close(y[:, -1], torch.tensor([[3.6, 4.]]))
        self.assertAlmostEqual(receipt[0]['relative_norm'], .12, places=6)
        torch.testing.assert_close(adapter.layers[0](h), h)

    def test_coefficient_is_divided_across_selected_layers(self):
        adapter = self.model()
        v = torch.tensor([[1., 0.]] * 4)
        with adapter.intervention(v, [1, 2, 3, 4], .12, 'sum_relative_l2') as receipt:
            h = torch.tensor([[[3., 4.]]])
            for layer in adapter.layers:
                h = layer(h)
        self.assertEqual(len(receipt), 4)
        self.assertTrue(all(abs(r['relative_norm'] - .03) < 1e-6 for r in receipt))

    def test_hook_cleanup_on_exception(self):
        adapter = self.model()
        v = torch.tensor([[1., 0.]] * 4)
        with self.assertRaises(RuntimeError):
            with adapter.intervention(v, [1, 2], .12):
                raise RuntimeError('synthetic interruption')
        self.assertTrue(all(not layer._forward_hooks for layer in adapter.layers))


if __name__ == '__main__':
    unittest.main()
