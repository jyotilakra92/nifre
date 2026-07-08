"""Closed-loop auto-tuning controller."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

from autotune.classifier import WorkloadClassifier
from autotune.policy import TuningGoal, TuningPolicy
from autotune.workload import workload_snapshot_from_metrics
from inference.data_model import EngineConfig

if TYPE_CHECKING:
    from inference.engine import Engine
    from inference.engine_worker import EngineWorker
    from observability import Observability


@dataclass(frozen=True)
class ControllerConfig:
    interval_sec: float = 30.0
    evaluation_sec: float = 60.0
    cooldown_sec: float = 30.0
    min_completed_requests: int = 1
    latency_regression_threshold_pct: float = 5.0
    throughput_regression_threshold_pct: float = 5.0
    error_rate_regression_delta: float = 0.02


@dataclass(frozen=True)
class EvaluationSnapshot:
    ttft_p95_ms: float
    tokens_per_sec: float
    error_rate: float


@dataclass
class PendingAttempt:
    name: str
    reason: str
    previous_config: EngineConfig
    baseline: EvaluationSnapshot
    started_at: float
    changes: dict[str, object]


@dataclass
class TuningStatus:
    enabled: bool
    goal: str
    pending_attempt: Optional[str] = None
    last_action: Optional[str] = None
    last_reason: Optional[str] = None


def _evaluation_from_metrics(snapshot: dict) -> EvaluationSnapshot:
    workload = workload_snapshot_from_metrics(snapshot)
    return EvaluationSnapshot(
        ttft_p95_ms=workload.ttft_p95_ms,
        tokens_per_sec=workload.tokens_per_sec,
        error_rate=workload.error_rate,
    )


def _attempt_name(changes: dict[str, object]) -> str:
    keys = ",".join(sorted(changes))
    return f"autotune:{keys}"


@dataclass
class TuningController:
    """Observe metrics, propose config changes, evaluate, promote or rollback."""

    engine: "Engine"
    observability: "Observability"
    goal: TuningGoal | str = TuningGoal.BALANCED
    worker: Optional["EngineWorker"] = None
    config: ControllerConfig = field(default_factory=ControllerConfig)
    classifier: WorkloadClassifier = field(default_factory=WorkloadClassifier)
    policy: TuningPolicy = field(default_factory=TuningPolicy)
    _time: Callable[[], float] = field(default=time.time, repr=False)
    _pending: Optional[PendingAttempt] = field(default=None, init=False, repr=False)
    _last_attempt_at: float = field(default=0.0, init=False, repr=False)
    _last_action: Optional[str] = field(default=None, init=False, repr=False)
    _last_reason: Optional[str] = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            self._enabled = True
            return
        if isinstance(self.goal, str):
            self.goal = TuningGoal(self.goal)
        self._stop.clear()
        self._enabled = True
        self._thread = threading.Thread(target=self._loop, name="tuning-controller", daemon=True)
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._enabled = False
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self.start()
        else:
            self.stop()

    def set_goal(self, goal: TuningGoal | str) -> None:
        self.goal = TuningGoal(goal) if isinstance(goal, str) else goal

    def snapshot(self) -> dict:
        """Full tuning state for admin API and dashboard."""
        status = self.status()
        engine_config = self.engine.get_config()
        pending = None
        if self._pending is not None:
            pending = {
                "name": self._pending.name,
                "reason": self._pending.reason,
                "started_at": self._pending.started_at,
                "changes": dict(self._pending.changes),
            }
        return {
            "available": True,
            "enabled": status.enabled,
            "goal": status.goal,
            "pending_attempt": status.pending_attempt,
            "pending": pending,
            "last_action": status.last_action,
            "last_reason": status.last_reason,
            "engine_config": {
                "max_concurrent_requests": engine_config.max_concurrent_requests,
                "prefill_chunk_size": engine_config.prefill_chunk_size,
                "max_tokens_per_step": engine_config.max_tokens_per_step,
                "use_paged_kv_cache": engine_config.use_paged_kv_cache,
                "use_prefix_cache": engine_config.use_prefix_cache,
            },
            "controller_config": {
                "interval_sec": self.config.interval_sec,
                "evaluation_sec": self.config.evaluation_sec,
                "cooldown_sec": self.config.cooldown_sec,
            },
        }

    def status(self) -> TuningStatus:
        pending = self._pending.name if self._pending else None
        goal_value = self.goal.value if isinstance(self.goal, TuningGoal) else str(self.goal)
        return TuningStatus(
            enabled=self._enabled,
            goal=goal_value,
            pending_attempt=pending,
            last_action=self._last_action,
            last_reason=self._last_reason,
        )

    def tick(self, now: Optional[float] = None) -> Optional[str]:
        """Run one controller cycle synchronously (for tests and manual driving)."""
        current_time = self._time() if now is None else now

        if self._pending is not None:
            if current_time - self._pending.started_at >= self.config.evaluation_sec:
                return self._finish_evaluation(current_time)
            return None

        if current_time - self._last_attempt_at < self.config.cooldown_sec:
            return None

        return self._maybe_start_attempt(current_time)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                self._last_action = "error"
                self._last_reason = str(exc)
            self._stop.wait(self.config.interval_sec)

    def _maybe_start_attempt(self, now: float) -> Optional[str]:
        metrics = self.observability.snapshot()
        completed = int(metrics.get("request_health", {}).get("completed_requests", 0))
        if completed < self.config.min_completed_requests:
            return None

        workload = workload_snapshot_from_metrics(metrics)
        labels = self.classifier.observe(workload)
        current = self.engine.get_config()
        plan = self.policy.propose(
            current,
            labels,
            self.goal,
            cache_initialized=self.engine.cache is not None,
        )
        if plan is None:
            return None

        name = _attempt_name(plan.changes)
        self.observability.optimization.record_attempt(name, details=plan.reason)
        self._apply_changes(plan.changes)

        self._pending = PendingAttempt(
            name=name,
            reason=plan.reason,
            previous_config=current,
            baseline=_evaluation_from_metrics(metrics),
            started_at=now,
            changes=dict(plan.changes),
        )
        self._last_attempt_at = now
        self._last_action = "attempted"
        self._last_reason = plan.reason
        return "attempted"

    def _finish_evaluation(self, now: float) -> str:
        assert self._pending is not None
        attempt = self._pending
        metrics = self.observability.snapshot()
        current = _evaluation_from_metrics(metrics)

        if self._is_regression(attempt.baseline, current):
            self._rollback(attempt)
            self.observability.optimization.record_rollback(
                attempt.name,
                details=f"regression vs baseline: {attempt.reason}",
            )
            self._last_action = "rolled_back"
            self._last_reason = attempt.reason
            action = "rolled_back"
        elif self._is_improvement(attempt.baseline, current):
            self.observability.optimization.record_promotion(
                attempt.name,
                details=attempt.reason,
            )
            self._last_action = "promoted"
            self._last_reason = attempt.reason
            action = "promoted"
        else:
            self._rollback(attempt)
            self.observability.optimization.record_rollback(
                attempt.name,
                details=f"neutral result, reverted: {attempt.reason}",
            )
            self._last_action = "rolled_back"
            self._last_reason = attempt.reason
            action = "rolled_back"

        self._pending = None
        self._last_attempt_at = now
        return action

    def _is_improvement(
        self,
        baseline: EvaluationSnapshot,
        current: EvaluationSnapshot,
    ) -> bool:
        goal = self.goal if isinstance(self.goal, TuningGoal) else TuningGoal(self.goal)

        if current.error_rate > baseline.error_rate + self.config.error_rate_regression_delta:
            return False

        if goal is TuningGoal.LATENCY:
            if baseline.ttft_p95_ms <= 0:
                return False
            return current.ttft_p95_ms < baseline.ttft_p95_ms

        if goal is TuningGoal.THROUGHPUT:
            if baseline.tokens_per_sec <= 0:
                return current.tokens_per_sec > 0
            threshold = 1.0 + self.config.throughput_regression_threshold_pct / 100.0
            return current.tokens_per_sec >= baseline.tokens_per_sec * threshold

        latency_ok = (
            baseline.ttft_p95_ms <= 0
            or current.ttft_p95_ms <= baseline.ttft_p95_ms * 1.02
        )
        throughput_ok = (
            baseline.tokens_per_sec <= 0
            or current.tokens_per_sec >= baseline.tokens_per_sec * 0.98
        )
        return latency_ok and throughput_ok and (
            current.ttft_p95_ms < baseline.ttft_p95_ms
            or current.tokens_per_sec > baseline.tokens_per_sec
        )

    def _is_regression(
        self,
        baseline: EvaluationSnapshot,
        current: EvaluationSnapshot,
    ) -> bool:
        if current.error_rate > baseline.error_rate + self.config.error_rate_regression_delta:
            return True

        goal = self.goal if isinstance(self.goal, TuningGoal) else TuningGoal(self.goal)

        if goal is TuningGoal.LATENCY:
            if baseline.ttft_p95_ms <= 0:
                return False
            threshold = 1.0 + self.config.latency_regression_threshold_pct / 100.0
            return current.ttft_p95_ms > baseline.ttft_p95_ms * threshold

        if goal is TuningGoal.THROUGHPUT:
            if baseline.tokens_per_sec <= 0:
                return False
            threshold = 1.0 - self.config.throughput_regression_threshold_pct / 100.0
            return current.tokens_per_sec < baseline.tokens_per_sec * threshold

        latency_regressed = (
            baseline.ttft_p95_ms > 0
            and current.ttft_p95_ms
            > baseline.ttft_p95_ms * (1.0 + self.config.latency_regression_threshold_pct / 100.0)
        )
        throughput_regressed = (
            baseline.tokens_per_sec > 0
            and current.tokens_per_sec
            < baseline.tokens_per_sec * (1.0 - self.config.throughput_regression_threshold_pct / 100.0)
        )
        return latency_regressed or throughput_regressed

    def _apply_changes(self, changes: dict[str, object]) -> None:
        if self.worker is not None:
            self.worker.reconfigure(**changes)
        else:
            self.engine.reconfigure(**changes)

    def _rollback(self, attempt: PendingAttempt) -> None:
        rollback_kwargs = {
            key: getattr(attempt.previous_config, key) for key in attempt.changes
        }
        self._apply_changes(rollback_kwargs)
