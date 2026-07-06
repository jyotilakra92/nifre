"""Rolling metrics storage and snapshot assembly for the observability dashboard."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def latency_summary(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": round(percentile(samples, 50) * 1000, 2),
        "p95_ms": round(percentile(samples, 95) * 1000, 2),
        "p99_ms": round(percentile(samples, 99) * 1000, 2),
        "mean_ms": round(sum(samples) / len(samples) * 1000, 2),
    }


@dataclass
class TimeSeries:
    maxlen: int = 120

    def __post_init__(self) -> None:
        self.points: Deque[Tuple[float, float]] = deque(maxlen=self.maxlen)

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        self.points.append((timestamp or time.time(), value))

    def as_list(self) -> List[List[float]]:
        return [[ts, val] for ts, val in self.points]


class MetricsStore:
    """Thread-safe in-memory metrics for the inference engine dashboard."""

    def __init__(self, window_sec: float = 60.0) -> None:
        self.window_sec = window_sec
        self._lock = threading.Lock()

        self.total_enqueued = 0
        self.total_completed = 0
        self.total_errors = 0
        self.total_timeouts = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_decode_iterations = 0

        self._completion_timestamps: Deque[float] = deque(maxlen=5000)
        self._ttft_samples: Deque[float] = deque(maxlen=2000)
        self._total_latency_samples: Deque[float] = deque(maxlen=2000)
        self._prefill_step_samples: Deque[float] = deque(maxlen=2000)
        self._decode_step_samples: Deque[float] = deque(maxlen=2000)
        self._inter_token_samples: Deque[float] = deque(maxlen=5000)

        self.ts_requests_per_sec = TimeSeries()
        self.ts_active_requests = TimeSeries()
        self.ts_queued_requests = TimeSeries()
        self.ts_tokens_per_sec = TimeSeries()
        self.ts_input_tokens_per_sec = TimeSeries()
        self.ts_output_tokens_per_sec = TimeSeries()
        self.ts_batch_size = TimeSeries()
        self.ts_decode_iterations_per_sec = TimeSeries()
        self.ts_gpu_utilization = TimeSeries()
        self.ts_gpu_memory_gb = TimeSeries()
        self.ts_kv_cache_utilization = TimeSeries()

        self._token_events: Deque[Tuple[float, int, int]] = deque(maxlen=10000)
        self._decode_iteration_timestamps: Deque[float] = deque(maxlen=5000)

    def record_enqueue(self) -> None:
        with self._lock:
            self.total_enqueued += 1

    def record_completion(
        self,
        *,
        ttft_sec: Optional[float],
        total_latency_sec: float,
        input_tokens: int,
        output_tokens: int,
        status: str,
    ) -> None:
        now = time.time()
        with self._lock:
            if status == "timeout":
                self.total_timeouts += 1
            elif status == "error":
                self.total_errors += 1
            else:
                self.total_completed += 1

            self._completion_timestamps.append(now)
            self._total_latency_samples.append(total_latency_sec)
            if ttft_sec is not None:
                self._ttft_samples.append(ttft_sec)

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self._token_events.append((now, input_tokens, output_tokens))

    def record_prefill_step(self, duration_sec: float, batch_size: int) -> None:
        with self._lock:
            self._prefill_step_samples.append(duration_sec)
            self.ts_batch_size.add(float(batch_size))

    def record_decode_step(self, duration_sec: float, batch_size: int) -> None:
        now = time.time()
        with self._lock:
            self._decode_step_samples.append(duration_sec)
            self.ts_batch_size.add(float(batch_size))
            self.total_decode_iterations += 1
            self._decode_iteration_timestamps.append(now)

    def record_inter_token(self, gap_sec: float) -> None:
        with self._lock:
            self._inter_token_samples.append(gap_sec)

    def record_runtime_sample(
        self,
        *,
        gpu_utilization: float,
        gpu_memory_gb: float,
        kv_cache_utilization: float,
    ) -> None:
        with self._lock:
            self.ts_gpu_utilization.add(gpu_utilization)
            self.ts_gpu_memory_gb.add(gpu_memory_gb)
            self.ts_kv_cache_utilization.add(kv_cache_utilization)

    def _rate(self, timestamps: Deque[float]) -> float:
        cutoff = time.time() - self.window_sec
        count = sum(1 for ts in timestamps if ts >= cutoff)
        return count / self.window_sec if self.window_sec else 0.0

    def _token_rates(self) -> Tuple[float, float, float]:
        cutoff = time.time() - self.window_sec
        input_tokens = 0
        output_tokens = 0
        for ts, inp, out in self._token_events:
            if ts >= cutoff:
                input_tokens += inp
                output_tokens += out
        total = input_tokens + output_tokens
        window = self.window_sec or 1.0
        return total / window, input_tokens / window, output_tokens / window

    def snapshot(
        self,
        *,
        active_requests: int,
        queued_requests: int,
        runtime_info: Dict[str, object],
        optimization_info: Dict[str, object],
    ) -> Dict[str, object]:
        with self._lock:
            requests_per_sec = self._rate(self._completion_timestamps)
            decode_iters_per_sec = self._rate(self._decode_iteration_timestamps)
            tokens_per_sec, input_tps, output_tps = self._token_rates()

            finished_total = self.total_completed + self.total_errors + self.total_timeouts
            error_rate = self.total_errors / finished_total if finished_total else 0.0
            timeout_rate = self.total_timeouts / finished_total if finished_total else 0.0

            avg_tokens_per_request = (
                (self.total_input_tokens + self.total_output_tokens) / self.total_completed
                if self.total_completed
                else 0.0
            )

            self.ts_requests_per_sec.add(requests_per_sec)
            self.ts_active_requests.add(float(active_requests))
            self.ts_queued_requests.add(float(queued_requests))
            self.ts_tokens_per_sec.add(tokens_per_sec)
            self.ts_input_tokens_per_sec.add(input_tps)
            self.ts_output_tokens_per_sec.add(output_tps)
            self.ts_decode_iterations_per_sec.add(decode_iters_per_sec)

            ttft_list = list(self._ttft_samples)
            total_lat_list = list(self._total_latency_samples)
            prefill_list = list(self._prefill_step_samples)
            decode_list = list(self._decode_step_samples)
            inter_list = list(self._inter_token_samples)

        return {
            "timestamp": time.time(),
            "request_health": {
                "requests_per_sec": round(requests_per_sec, 3),
                "active_requests": active_requests,
                "queued_requests": queued_requests,
                "completed_requests": self.total_completed,
                "error_rate": round(error_rate, 4),
                "timeout_rate": round(timeout_rate, 4),
            },
            "latency": {
                "ttft": latency_summary(ttft_list),
                "total_request_latency": latency_summary(total_lat_list),
                "prefill_step_latency": latency_summary(prefill_list),
                "decode_step_latency": latency_summary(decode_list),
                "inter_token_latency": latency_summary(inter_list),
            },
            "throughput": {
                "tokens_per_sec": round(tokens_per_sec, 2),
                "input_tokens_per_sec": round(input_tps, 2),
                "output_tokens_per_sec": round(output_tps, 2),
                "tokens_per_request": round(avg_tokens_per_request, 2),
                "decode_iterations_per_sec": round(decode_iters_per_sec, 2),
            },
            "gpu_runtime": runtime_info,
            "optimization_history": optimization_info,
            "timeseries": {
                "requests_per_sec": self.ts_requests_per_sec.as_list(),
                "active_requests": self.ts_active_requests.as_list(),
                "queued_requests": self.ts_queued_requests.as_list(),
                "tokens_per_sec": self.ts_tokens_per_sec.as_list(),
                "input_tokens_per_sec": self.ts_input_tokens_per_sec.as_list(),
                "output_tokens_per_sec": self.ts_output_tokens_per_sec.as_list(),
                "batch_size": self.ts_batch_size.as_list(),
                "decode_iterations_per_sec": self.ts_decode_iterations_per_sec.as_list(),
                "gpu_utilization": self.ts_gpu_utilization.as_list(),
                "gpu_memory_gb": self.ts_gpu_memory_gb.as_list(),
                "kv_cache_utilization": self.ts_kv_cache_utilization.as_list(),
            },
        }
