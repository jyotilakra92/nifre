"""Deterministic tuning policy: workload labels → proposed engine config."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from autotune.classifier import WorkloadLabels
from inference.data_model import EngineConfig


class TuningGoal(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    BALANCED = "balanced"


@dataclass(frozen=True)
class PolicyConfig:
    min_prefill_chunk_size: int = 32
    max_prefill_chunk_size: int = 512
    prefill_chunk_step: int = 64

    min_max_tokens_per_step: int = 256
    max_max_tokens_per_step: int = 4096
    token_budget_step: int = 256

    min_max_concurrent_requests: int = 1
    max_max_concurrent_requests: int = 16
    concurrency_step: int = 1


@dataclass(frozen=True)
class TuningPlan:
    config: EngineConfig
    reason: str
    changes: dict[str, object] = field(default_factory=dict)


def _step_int(
    current: int,
    delta: int,
    *,
    minimum: int,
    maximum: int,
) -> int | None:
    if delta == 0:
        return None
    new_value = max(minimum, min(maximum, current + delta))
    return new_value if new_value != current else None


def _apply_updates(current: EngineConfig, updates: dict[str, object]) -> EngineConfig:
    return replace(current, **updates)


@dataclass
class TuningPolicy:
    """Rule-based mapping from workload labels to engine configuration changes."""

    config: PolicyConfig = field(default_factory=PolicyConfig)

    def propose(
        self,
        current: EngineConfig,
        labels: WorkloadLabels,
        goal: TuningGoal | str,
        *,
        cache_initialized: bool = False,
    ) -> TuningPlan | None:
        if labels.error_elevated:
            return None

        if isinstance(goal, str):
            goal = TuningGoal(goal)

        updates: dict[str, object] = {}
        reasons: list[str] = []

        if goal is TuningGoal.LATENCY:
            self._apply_latency_rules(current, labels, updates, reasons)
        elif goal is TuningGoal.THROUGHPUT:
            self._apply_throughput_rules(current, labels, updates, reasons, cache_initialized)
        else:
            self._apply_balanced_rules(current, labels, updates, reasons, cache_initialized)

        self._apply_prefix_rules(current, labels, updates, reasons, cache_initialized)

        if not updates:
            return None

        return TuningPlan(
            config=_apply_updates(current, updates),
            reason="; ".join(reasons),
            changes=dict(updates),
        )

    def _apply_latency_rules(
        self,
        current: EngineConfig,
        labels: WorkloadLabels,
        updates: dict[str, object],
        reasons: list[str],
    ) -> None:
        if labels.latency_sensitive or labels.queue_high:
            new_chunk = _step_int(
                current.prefill_chunk_size,
                -self.config.prefill_chunk_step,
                minimum=self.config.min_prefill_chunk_size,
                maximum=self.config.max_prefill_chunk_size,
            )
            if new_chunk is not None:
                updates["prefill_chunk_size"] = new_chunk
                reasons.append("reduce prefill chunk size for lower TTFT")

        if labels.latency_sensitive:
            new_budget = _step_int(
                current.max_tokens_per_step,
                -self.config.token_budget_step,
                minimum=self.config.min_max_tokens_per_step,
                maximum=self.config.max_max_tokens_per_step,
            )
            if new_budget is not None:
                updates["max_tokens_per_step"] = new_budget
                reasons.append("lower token budget to interleave decode sooner")

    def _apply_throughput_rules(
        self,
        current: EngineConfig,
        labels: WorkloadLabels,
        updates: dict[str, object],
        reasons: list[str],
        cache_initialized: bool,
    ) -> None:
        if labels.queue_high and not cache_initialized:
            new_concurrency = _step_int(
                current.max_concurrent_requests,
                self.config.concurrency_step,
                minimum=self.config.min_max_concurrent_requests,
                maximum=self.config.max_max_concurrent_requests,
            )
            if new_concurrency is not None:
                updates["max_concurrent_requests"] = new_concurrency
                reasons.append("raise concurrency to drain queue")

        if labels.throughput_low:
            new_budget = _step_int(
                current.max_tokens_per_step,
                self.config.token_budget_step,
                minimum=self.config.min_max_tokens_per_step,
                maximum=self.config.max_max_tokens_per_step,
            )
            if new_budget is not None:
                updates["max_tokens_per_step"] = new_budget
                reasons.append("raise token budget for higher throughput")

    def _apply_balanced_rules(
        self,
        current: EngineConfig,
        labels: WorkloadLabels,
        updates: dict[str, object],
        reasons: list[str],
        cache_initialized: bool,
    ) -> None:
        if labels.latency_sensitive:
            new_chunk = _step_int(
                current.prefill_chunk_size,
                -self.config.prefill_chunk_step // 2,
                minimum=self.config.min_prefill_chunk_size,
                maximum=self.config.max_prefill_chunk_size,
            )
            if new_chunk is not None:
                updates["prefill_chunk_size"] = new_chunk
                reasons.append("trim prefill chunks to protect latency")

        if labels.throughput_low:
            half_step = max(self.config.token_budget_step // 2, 1)
            new_budget = _step_int(
                current.max_tokens_per_step,
                half_step,
                minimum=self.config.min_max_tokens_per_step,
                maximum=self.config.max_max_tokens_per_step,
            )
            if new_budget is not None:
                updates["max_tokens_per_step"] = new_budget
                reasons.append("nudge token budget up for throughput")

        if labels.queue_high:
            if not cache_initialized:
                new_concurrency = _step_int(
                    current.max_concurrent_requests,
                    self.config.concurrency_step,
                    minimum=self.config.min_max_concurrent_requests,
                    maximum=self.config.max_max_concurrent_requests,
                )
                if new_concurrency is not None:
                    updates["max_concurrent_requests"] = new_concurrency
                    reasons.append("raise concurrency to reduce queue wait")
            else:
                new_chunk = _step_int(
                    current.prefill_chunk_size,
                    -self.config.prefill_chunk_step // 2,
                    minimum=self.config.min_prefill_chunk_size,
                    maximum=self.config.max_prefill_chunk_size,
                )
                if new_chunk is not None and "prefill_chunk_size" not in updates:
                    updates["prefill_chunk_size"] = new_chunk
                    reasons.append("reduce prefill chunks while queue is high")

    def _apply_prefix_rules(
        self,
        current: EngineConfig,
        labels: WorkloadLabels,
        updates: dict[str, object],
        reasons: list[str],
        cache_initialized: bool,
    ) -> None:
        if (
            labels.prefix_friendly
            and not current.use_prefix_cache
            and current.use_paged_kv_cache
            and not cache_initialized
        ):
            updates["use_prefix_cache"] = True
            reasons.append("enable prefix cache for shared-prefix workload")
