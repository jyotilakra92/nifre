import sys
import uuid
from pathlib import Path
from typing import List

import torch

from data_model import InferenceRequest
from scheduler import Scheduler

_TEST_LLM = Path(__file__).resolve().parent / "test-llm"
if str(_TEST_LLM) not in sys.path:
    sys.path.insert(0, str(_TEST_LLM))

from generate import batch_token_ids, make_kv_cache  # noqa: E402
from sampler import sample_greedy  # noqa: E402


class Engine:
    """Continuous-batching inference engine: model + KV cache + scheduler."""

    def __init__(self, model, max_concurrent_requests: int, device: torch.device):
        self.model = model
        self.device = device
        self.max_concurrent_requests = max_concurrent_requests
        self.scheduler = Scheduler(max_concurrent_requests)
        self.cache = None

    def add_request(self, prompt_token_ids: List[int], max_new_tokens: int) -> str:
        request_id = uuid.uuid4().hex[:8]
        request = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
        )
        self.scheduler.add_request(request)
        return request_id

    def get_completed(self) -> dict:
        return self.scheduler.completed

    def step(self) -> bool:
        if not self.scheduler.has_work():
            return False

        self._ensure_cache()
        result = self.scheduler.schedule()

        if result.prefill_requests:
            self._run_prefill(result.prefill_requests)

        if result.decode_requests:
            self._run_decode(result.decode_requests)

        return True

    def run_until_done(self) -> None:
        while self.step():
            pass

    def _ensure_cache(self) -> None:
        if self.cache is None:
            self.cache = make_kv_cache(self.model, self.device)
            self.cache.init_batch(self.max_concurrent_requests)

    def _run_prefill(self, requests: List[InferenceRequest]) -> None:
        for request in requests:
            self.cache.reset_slot(request.batch_idx)

        token_lists = [request.prompt_token_ids for request in requests]
        token_ids, input_lens = batch_token_ids(token_lists, self.device)
        cache_batch_indices = [request.batch_idx for request in requests]

        logits = self.model(
            token_ids,
            kv_cache=self.cache,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )

        for i, request in enumerate(requests):
            self.scheduler.mark_prefill_done(request)
            next_token = sample_greedy(logits[i : i + 1, -1, :]).item()
            self.scheduler.mark_decode_done(request, next_token)

    def _run_decode(self, requests: List[InferenceRequest]) -> None:
        tokens = [[request.output_token_ids[-1]] for request in requests]
        token_ids = torch.tensor(tokens, device=self.device)
        cache_batch_indices = [request.batch_idx for request in requests]

        logits = self.model(
            token_ids,
            kv_cache=self.cache,
            cache_batch_indices=cache_batch_indices,
        )

        for i, request in enumerate(requests):
            next_token = sample_greedy(logits[i : i + 1, -1, :]).item()
            self.scheduler.mark_decode_done(request, next_token)


def _smoke_test():
    from gpt_model import GptModel, GPT_CONFIG_124M

    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    model = GptModel(cfg).to(device).eval()

    engine = Engine(model, max_concurrent_requests=2, device=device)
    engine.add_request([1, 2, 3], max_new_tokens=2)
    engine.add_request([10, 20, 30, 40], max_new_tokens=2)
    engine.add_request([5, 6], max_new_tokens=2)
    engine.run_until_done()

    assert len(engine.get_completed()) == 3
    for request in engine.get_completed().values():
        assert request.num_generated == 2

    print("engine smoke test passed")


if __name__ == "__main__":
    _smoke_test()
