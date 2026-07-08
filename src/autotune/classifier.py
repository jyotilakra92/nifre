"""Rule-based workload classifier for auto-tuning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

from autotune.workload import WorkloadSnapshot


@dataclass(frozen=True)
class ClassifierConfig:
    queue_high_threshold: int = 2
    queue_high_window_sec: float = 60.0
    queue_high_min_samples: int = 3
    ttft_p95_latency_sensitive_ms: float = 500.0
    prefix_friendly_hit_rate: float = 0.30
    throughput_low_tps: float = 10.0
    error_elevated_rate: float = 0.05


@dataclass(frozen=True)
class WorkloadLabels:
    queue_high: bool = False
    prefix_friendly: bool = False
    latency_sensitive: bool = False
    throughput_low: bool = False
    error_elevated: bool = False

    def active(self) -> frozenset[str]:
        labels = []
        if self.queue_high:
            labels.append("queue_high")
        if self.prefix_friendly:
            labels.append("prefix_friendly")
        if self.latency_sensitive:
            labels.append("latency_sensitive")
        if self.throughput_low:
            labels.append("throughput_low")
        if self.error_elevated:
            labels.append("error_elevated")
        return frozenset(labels)


@dataclass
class WorkloadClassifier:
    """Maps rolling observability snapshots to workload labels."""

    config: ClassifierConfig = field(default_factory=ClassifierConfig)
    _queue_samples: Deque[Tuple[float, int]] = field(default_factory=deque, init=False, repr=False)

    def observe(self, snapshot: WorkloadSnapshot) -> WorkloadLabels:
        """Record queue depth and return labels for this snapshot."""
        self._queue_samples.append((snapshot.timestamp, snapshot.queued_requests))
        self._prune_queue_samples(snapshot.timestamp)
        return self._classify(snapshot)

    def classify(self, snapshot: WorkloadSnapshot) -> WorkloadLabels:
        """Classify without recording queue history (for unit tests)."""
        return self._classify(snapshot)

    def _classify(self, snapshot: WorkloadSnapshot) -> WorkloadLabels:
        return WorkloadLabels(
            queue_high=self._queue_sustained_high(),
            prefix_friendly=snapshot.prefix_cache_hit_rate >= self.config.prefix_friendly_hit_rate,
            latency_sensitive=(
                snapshot.ttft_p95_ms >= self.config.ttft_p95_latency_sensitive_ms
                and snapshot.ttft_p95_ms > 0
            ),
            throughput_low=(
                snapshot.tokens_per_sec < self.config.throughput_low_tps
                and (snapshot.active_requests > 0 or snapshot.queued_requests > 0)
            ),
            error_elevated=snapshot.error_rate >= self.config.error_elevated_rate,
        )

    def _prune_queue_samples(self, now: float) -> None:
        cutoff = now - self.config.queue_high_window_sec
        while self._queue_samples and self._queue_samples[0][0] < cutoff:
            self._queue_samples.popleft()

    def _queue_sustained_high(self) -> bool:
        recent = list(self._queue_samples)
        if len(recent) < self.config.queue_high_min_samples:
            return False

        if not all(
            queued >= self.config.queue_high_threshold for _, queued in recent
        ):
            return False

        window_span = recent[-1][0] - recent[0][0]
        min_span = self.config.queue_high_window_sec * 0.9
        return window_span >= min_span
