"""Continuous-batching inference engine: model + KV cache + scheduler."""

import time
import uuid
from queue import Empty, Queue
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional

import torch

from inference.batching import make_kv_cache
from inference.data_model import InferenceRequest, RequestState
from inference.model_interface import InferenceModel
from inference.model_runner import ModelRunner
from inference.scheduler import Scheduler

if TYPE_CHECKING:
    from observability.collector import MetricsCollector

TokenCallback = Callable[[int], None]


class Engine:
    """Continuous-batching inference engine: model + KV cache + scheduler."""

    def __init__(
        self,
        model: InferenceModel,
        max_concurrent_requests: int,
        device: torch.device,
        metrics_collector: Optional["MetricsCollector"] = None,
        prefill_chunk_size: int = 128,
        max_tokens_per_step: int = 2048,
    ):
        self.model = model
        self.device = device
        self.max_concurrent_requests = max_concurrent_requests
        self.scheduler = Scheduler(max_concurrent_requests, max_tokens_per_step)
        self.model_runner = ModelRunner(model, device)
        self.cache = None
        self.metrics = metrics_collector
        self.prefill_chunk_size = prefill_chunk_size
        self.max_tokens_per_step = max_tokens_per_step
        self._token_callbacks: dict[str, TokenCallback] = {}

    def add_request(self, prompt_token_ids: List[int], max_new_tokens: int) -> str:
        request_id = uuid.uuid4().hex[:8]
        request = InferenceRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            prefill_chunk_size=self.prefill_chunk_size,
        )
        self.scheduler.add_request(request)
        if self.metrics:
            self.metrics.on_request_enqueued()
        return request_id

    def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
    ) -> Iterator[int]:
        """Yield output token ids as they are produced (streaming counterpart to ``generate``)."""
        request_id = self.add_request(prompt_token_ids, max_new_tokens)
        queue: Queue[int] = Queue()
        self.register_token_callback(request_id, queue.put)

        try:
            while (
                request_id not in self.scheduler.completed
                and request_id not in self.scheduler.failed
            ):
                self.step()
                while True:
                    try:
                        yield queue.get_nowait()
                    except Empty:
                        break
        finally:
            self.unregister_token_callback(request_id)

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

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> InferenceRequest:
        """Queue one request and block until it finishes (non-streaming counterpart to ``generate_stream``)."""
        request_id = self.add_request(prompt_token_ids, max_new_tokens)
        deadline = time.time() + timeout_seconds if timeout_seconds else None

        try:
            while (
                request_id not in self.scheduler.completed
                and request_id not in self.scheduler.failed
            ):
                if deadline and time.time() > deadline:
                    request = self._cancel_request(request_id, status="timeout")
                    if self.metrics and request is not None:
                        self.metrics.on_request_failed(request, "timeout")
                    return self.scheduler.failed[request_id]

                self.step()
        finally:
            self.unregister_token_callback(request_id)

        if request_id in self.scheduler.completed:
            return self.scheduler.completed[request_id]
        return self.scheduler.failed[request_id]

    def register_token_callback(self, request_id: str, callback: TokenCallback) -> None:
        """Register a per-request hook invoked after each generated token."""
        self._token_callbacks[request_id] = callback

    def unregister_token_callback(self, request_id: str) -> None:
        self._token_callbacks.pop(request_id, None)

    def _cancel_request(self, request_id: str, status: str) -> Optional[InferenceRequest]:
        request = self.scheduler.cancel_request(request_id, status=status)
        return request

    def _ensure_cache(self) -> None:
        if self.cache is None:
            dtype = getattr(self.model, "dtype", torch.float16)
            self.cache = make_kv_cache(self.model.config, self.device, dtype=dtype)
            self.cache.init_batch(self.max_concurrent_requests)

    def _emit_token(self, request_id: str, token_id: int) -> None:
        callback = self._token_callbacks.get(request_id)
        if callback is not None:
            callback(token_id)

    def _run_prefill(self, requests: List[InferenceRequest]) -> None:
        start = time.perf_counter()
        token_ids = self.model_runner.prefill(self.cache, requests)
        duration = time.perf_counter() - start
        if self.metrics:
            self.metrics.on_prefill_batch(requests, duration)

        for request, token_id in zip(requests, token_ids):
            if token_id is None:
                continue
            self.scheduler.mark_prefill_done(request)
            self.scheduler.mark_decode_done(request, token_id)
            self._emit_token(request.request_id, token_id)
            if self.metrics:
                if request.state == RequestState.FINISHED:
                    self.metrics.on_request_finished(request)

    def _run_decode(self, requests: List[InferenceRequest]) -> None:
        start = time.perf_counter()
        token_ids = self.model_runner.decode(self.cache, requests)
        duration = time.perf_counter() - start
        if self.metrics:
            self.metrics.on_decode_batch(len(requests), duration)

        for request, token_id in zip(requests, token_ids):
            self.scheduler.mark_decode_done(request, token_id)
            self._emit_token(request.request_id, token_id)
            if self.metrics:
                self.metrics.on_decode_token(request)
                if request.state == RequestState.FINISHED:
                    self.metrics.on_request_finished(request)
