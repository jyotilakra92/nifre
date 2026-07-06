"""Hooks inference engine events into the metrics store."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from observability.metrics_store import MetricsStore
from observability.optimization import OptimizationTracker
from observability.runtime_probe import RuntimeProbe

if TYPE_CHECKING:
    from inference.data_model import InferenceRequest
    from inference.engine import Engine


class MetricsCollector:
    """Records per-request and per-step metrics from the inference engine."""

    def __init__(
        self,
        store: MetricsStore,
        runtime_probe: RuntimeProbe,
        optimization: OptimizationTracker,
    ) -> None:
        self.store = store
        self.runtime_probe = runtime_probe
        self.optimization = optimization
        self._engine: Optional["Engine"] = None

    def attach(self, engine: "Engine") -> None:
        self._engine = engine
        self.runtime_probe.attach(engine)

    def on_request_enqueued(self) -> None:
        self.store.record_enqueue()

    def on_prefill_batch(self, requests: list, duration_sec: float) -> None:
        self.store.record_prefill_step(duration_sec, len(requests))
        now = time.time()
        for request in requests:
            request.prefill_duration_sec = duration_sec
            request.first_token_at = now
            request.last_token_at = now

    def on_decode_batch(self, batch_size: int, duration_sec: float) -> None:
        self.store.record_decode_step(duration_sec, batch_size)

    def on_decode_token(self, request: "InferenceRequest") -> None:
        now = time.time()
        if request.last_token_at is not None and request.num_generated >= 1:
            self.store.record_inter_token(now - request.last_token_at)
        request.last_token_at = now

    def on_request_finished(self, request: "InferenceRequest") -> None:
        request.finished_at = time.time()
        ttft = None
        if request.first_token_at is not None:
            ttft = request.first_token_at - request.created_at
        total_latency = request.finished_at - request.created_at
        self.store.record_completion(
            ttft_sec=ttft,
            total_latency_sec=total_latency,
            input_tokens=request.num_prompt_tokens,
            output_tokens=request.num_generated,
            status=request.status,
        )
        self._maybe_update_optimization(total_latency, request)

    def on_request_failed(self, request: "InferenceRequest", status: str) -> None:
        request.status = status
        request.finished_at = time.time()
        self.store.record_completion(
            ttft_sec=None,
            total_latency_sec=request.finished_at - request.created_at,
            input_tokens=request.num_prompt_tokens,
            output_tokens=request.num_generated,
            status=status,
        )

    def _maybe_update_optimization(self, total_latency_sec: float, request: "InferenceRequest") -> None:
        latency_ms = total_latency_sec * 1000
        tokens = request.num_prompt_tokens + request.num_generated
        throughput = tokens / total_latency_sec if total_latency_sec > 0 else 0.0
        if self.optimization.baseline_latency_ms is None:
            self.optimization.set_baseline(latency_ms, throughput)
        else:
            self.optimization.update_current(latency_ms, throughput)

    def snapshot(self) -> dict:
        engine = self._engine
        active = len(engine.scheduler.running) if engine else 0
        queued = len(engine.scheduler.waiting) if engine else 0
        self.runtime_probe.sample_to_store(self.store)
        return self.store.snapshot(
            active_requests=active,
            queued_requests=queued,
            runtime_info=self.runtime_probe.snapshot(),
            optimization_info=self.optimization.snapshot(),
        )
