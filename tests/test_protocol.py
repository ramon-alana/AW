"""Synthetic regression fixtures only; no study images, scores, or vote records."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from aw.core import choose_window, direction
from aw.common import paired_seed
from aw.pipeline import image_ci, matrix, random_bands, stage, test_conditions, scan_conditions
from aw.human import holm, summarize


class ProtocolTests(unittest.TestCase):
    def test_response_window_uses_contiguous_mean_not_peak(self):
        gains = [0., 8., 0., 0., 4., 4., 4., 4., 0.]
        self.assertEqual(choose_window(gains, 4)['layers'], [5, 6, 7, 8])

    def test_window_tie_uses_earliest(self):
        self.assertEqual(choose_window([1.] * 8, 4)['layers'], [1, 2, 3, 4])

    def test_projection_removes_neutral_pc_and_is_unit(self):
        neutral = np.array([[-2., 0., 0.], [-1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        target = neutral + np.array([2., 3., 0.])
        vector, _ = direction(neutral, target, k=1)
        np.testing.assert_allclose(vector, [0., 1., 0.], atol=1e-6)

    def test_zero_direction_rejected(self):
        x = np.eye(4)
        with self.assertRaises(ValueError):
            direction(x, x)

    def test_paired_seed_stable_and_identity_specific(self):
        self.assertEqual(paired_seed('synthetic-a', 42), paired_seed('synthetic-a', 42))
        self.assertNotEqual(paired_seed('synthetic-a', 42), paired_seed('synthetic-b', 42))

    def test_random_bands_five_fixed_contiguous(self):
        for model, n in [('qwen', 28), ('phi', 32), ('minicpm', 36)]:
            bands = random_bands(model, 'joy', n)
            self.assertEqual(len({tuple(b) for b in bands}), 5)
            self.assertEqual(bands, random_bands(model, 'joy', n))
            for b in bands:
                self.assertEqual(b, list(range(b[0], b[0] + 4)))
                self.assertTrue(1 <= b[0] <= b[-1] <= n)

    def test_protocol_has_23_unique_conditions_for_every_model(self):
        root = Path(__file__).resolve().parents[1]
        for key in ['qwen', 'phi', 'minicpm']:
            cfg = json.loads((root / f'configs/{key}.json').read_text())
            selection = {'probe_layer': 2, 'windows': {e: {'layers': [5, 6, 7, 8], 'best_single': 6} for e in ['joy', 'sadness']}}
            plan = test_conditions(cfg, selection)
            self.assertEqual(len({c['name'] for c in plan}), 23)
            matched = next(c for c in plan if c['name'] == 'joy_window_sum_budget')
            self.assertEqual(matched['budget'], 'sum_relative_l2')
            self.assertFalse(next(c for c in plan if c['name'] == 'joy_window')['prompt'])
            self.assertEqual(next(c for c in plan if c['name'] == 'joy_prompt')['prompt'], 'joy')
            self.assertEqual(len(list(scan_conditions(cfg))), 3 + 2 * cfg['model']['layers'])

    def test_image_bootstrap_uses_image_count(self):
        result = image_ci([1., 1., 1., 1.], repeats=513)
        self.assertEqual(result['n_images'], 4)
        self.assertEqual((result['low'], result['high'], result['delta']), (1., 1., 1.))

    def test_bootstrap_reproducible(self):
        delta = [-.2, .3, .1, .7, -.1]
        self.assertEqual(image_ci(delta, repeats=700), image_ci(delta, repeats=700))

    def test_nan_inference_rejected(self):
        with self.assertRaises(ValueError):
            image_ci([0., np.nan])

    def test_complete_grid_required(self):
        stamp = {'config_sha256': 'test', 'manifest_sha256': 'test'}
        row = {'image_id': 'synthetic-a', 'condition': 'baseline', 'seed': 42,
               'inputs': stamp, 'partition': 'test', 'emotion_scores': {'joy': .5, 'sadness': .5}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'fixture.jsonl'
            p.write_text(json.dumps(row) + '\n')
            table = matrix([p], ['synthetic-a'], ['baseline'], [42], stamp, 'test')
            self.assertEqual(len(table), 1)
            with self.assertRaises(ValueError):
                matrix([p], ['synthetic-a', 'synthetic-b'], ['baseline'], [42], stamp, 'test')
            with self.assertRaises(ValueError):
                matrix([p, p], ['synthetic-a'], ['baseline'], [42], stamp, 'test')
            with self.assertRaises(ValueError):
                matrix([p], ['synthetic-a'], ['baseline'], [42], stamp, 'validation')

    def test_completed_stage_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with stage(Path(tmp), 'test', {'synthetic': True}) as out:
                (out / 'value.txt').write_text('synthetic')
            self.assertTrue((out / 'complete.json').exists())
            with self.assertRaises(FileExistsError):
                with stage(Path(tmp), 'test', {}):
                    pass

    def test_holm_monotonic_adjustment(self):
        np.testing.assert_allclose(holm([.04, .01, .03, .001]), [.06, .03, .06, .004])

    def test_net_margin_is_not_absolute_majority(self):
        rows = []
        for comparator in ['AW_vs_Neutral', 'AW_vs_Direct']:
            for tone in ['joy', 'sadness']:
                for i in range(3):
                    rows.append({'comparison': comparator, 'tone': tone, 'image_id': str(i), 'win': 2, 'tie': 2, 'loss': 1})
        for summary in summarize(rows):
            self.assertEqual(summary['positive_zero_negative_images'], [3, 0, 0])
            self.assertAlmostEqual(summary['margin']['delta'], .2)
            self.assertEqual(summary['sign_p'], .25)


if __name__ == '__main__':
    unittest.main()
