import uuid
from typing import List

import torch

from inference.batching import make_kv_cache
from inference.data_model import InferenceRequest
from inference.model_interface import InferenceModel
from inference.model_runner import ModelRunner
from inference.scheduler import Scheduler


class Engine:
    """Continuous-batching inference engine: model + KV cache + scheduler."""

    def __init__(
        self,
        model: InferenceModel,
        max_concurrent_requests: int,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.max_concurrent_requests = max_concurrent_requests
        self.scheduler = Scheduler(max_concurrent_requests)
        self.model_runner = ModelRunner(model, device)
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

    def generate(self, prompt_token_ids: List[int], max_new_tokens: int) -> InferenceRequest:
        """Queue one request and block until it finishes."""
        request_id = self.add_request(prompt_token_ids, max_new_tokens)
        while request_id not in self.scheduler.completed:
            self.step()
        return self.scheduler.completed[request_id]

    def _ensure_cache(self) -> None:
        if self.cache is None:
            dtype = getattr(self.model, "dtype", torch.float16)
            self.cache = make_kv_cache(self.model.config, self.device, dtype=dtype)
            self.cache.init_batch(self.max_concurrent_requests)

    def _run_prefill(self, requests: List[InferenceRequest]) -> None:
        token_ids = self.model_runner.prefill(self.cache, requests)
        for request, token_id in zip(requests, token_ids):
            self.scheduler.mark_prefill_done(request)
            self.scheduler.mark_decode_done(request, token_id)

    def _run_decode(self, requests: List[InferenceRequest]) -> None:
        token_ids = self.model_runner.decode(self.cache, requests)
        for request, token_id in zip(requests, token_ids):
            self.scheduler.mark_decode_done(request, token_id)
