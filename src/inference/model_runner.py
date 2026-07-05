from typing import List

import torch

from inference.data_model import InferenceRequest
from model.generate import batch_token_ids
from model.kv_cache import KVCache
from model.sampler import sample_greedy


class ModelRunner:
    """Runs prefill and decode forward passes against the model and KV cache."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def prefill(self, cache: KVCache, requests: List[InferenceRequest]) -> List[int]:
        for request in requests:
            cache.reset_slot(request.batch_idx)

        token_lists = [request.prompt_token_ids for request in requests]
        token_ids, input_lens = batch_token_ids(token_lists, self.device)
        cache_batch_indices = [request.batch_idx for request in requests]

        logits = self.model(
            token_ids,
            kv_cache=cache,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )

        return [
            sample_greedy(logits[i : i + 1, -1, :]).item()
            for i in range(len(requests))
        ]

    @torch.no_grad()
    def decode(self, cache: KVCache, requests: List[InferenceRequest]) -> List[int]:
        tokens = [[request.output_token_ids[-1]] for request in requests]
        token_ids = torch.tensor(tokens, device=self.device)
        cache_batch_indices = [request.batch_idx for request in requests]

        logits = self.model(
            token_ids,
            kv_cache=cache,
            cache_batch_indices=cache_batch_indices,
        )

        return [
            sample_greedy(logits[i : i + 1, -1, :]).item()
            for i in range(len(requests))
        ]
