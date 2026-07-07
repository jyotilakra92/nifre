from typing import List, Optional

import torch

from inference.batching import batch_token_ids
from inference.data_model import InferenceRequest
from inference.model_interface import InferenceModel
from model.kv_cache import KVCache
from model.sampler import sample_greedy


class ModelRunner:
    """Runs prefill and decode forward passes against the model and KV cache."""

    def __init__(self, model: InferenceModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def prefill(self, cache: KVCache, requests: List[InferenceRequest]) -> List[Optional[int]]:
        """Process one prefill chunk per request.

        Each request advances ``prefill_offset`` by the number of prompt tokens
        processed this step. Returns the first sampled token when the full prompt
        has been cached, otherwise ``None``.
        """
        for request in requests:
            if request.prefill_offset == 0:
                cache.reset_slot(request.batch_idx)

        token_lists = []
        for request in requests:
            start = request.prefill_offset
            end = min(start + request.prefill_chunk_size, request.num_prompt_tokens)
            token_lists.append(request.prompt_token_ids[start:end])

        token_ids, input_lens = batch_token_ids(
            token_lists,
            self.device,
            pad_id=self.model.config.pad_token_id,
        )
        cache_batch_indices = [request.batch_idx for request in requests]

        logits = self.model(
            token_ids,
            kv_cache=cache,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )

        results: List[Optional[int]] = []
        for i, request in enumerate(requests):
            chunk_len = input_lens[i].item()
            request.prefill_offset += chunk_len

            if request.prefill_complete:
                results.append(sample_greedy(logits[i : i + 1, -1, :]).item())
            else:
                results.append(None)

        return results

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
