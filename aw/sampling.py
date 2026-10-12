"""CPU nucleus prefix sum for strict deterministic PyTorch CUDA sampling."""
from contextlib import contextmanager
import torch

@contextmanager
def deterministic_nucleus():
    from transformers.generation.logits_process import TopPLogitsWarper
    original=TopPLogitsWarper.__call__
    def call(self,input_ids,scores):
        values,indices=torch.sort(scores,descending=False)
        cumulative=values.softmax(-1).cpu().cumsum(-1).to(scores.device)
        remove=cumulative<=(1-self.top_p);remove[...,-self.min_tokens_to_keep:]=False
        return scores.masked_fill(remove.scatter(1,indices,remove),self.filter_value)
    TopPLogitsWarper.__call__=call
    try:yield
    finally:TopPLogitsWarper.__call__=original
