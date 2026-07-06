"""Observability bundle: metrics store, collector, runtime probe, optimization tracker."""

from __future__ import annotations

from typing import Optional

from observability.collector import MetricsCollector
from observability.metrics_store import MetricsStore
from observability.optimization import OptimizationTracker
from observability.runtime_probe import RuntimeProbe


class Observability:
    """Wires together all observability components for the inference engine."""

    def __init__(
        self,
        *,
        model_name: str = "unknown",
        runtime: str = "custom",
        precision: Optional[str] = None,
    ) -> None:
        self.store = MetricsStore()
        self.optimization = OptimizationTracker()
        self.runtime_probe = RuntimeProbe(
            runtime=runtime,
            model_name=model_name,
            precision=precision,
        )
        self.collector = MetricsCollector(self.store, self.runtime_probe, self.optimization)

    def attach(self, engine) -> MetricsCollector:
        engine.metrics = self.collector
        self.collector.attach(engine)
        return self.collector

    def snapshot(self) -> dict:
        return self.collector.snapshot()
