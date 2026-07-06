"""Tracks optimization experiments and baseline vs current performance."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OptimizationEvent:
    name: str
    action: str
    timestamp: float
    details: str = ""


class OptimizationTracker:
    """Records optimization attempts and compares baseline vs current metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.baseline_latency_ms: Optional[float] = None
        self.current_latency_ms: Optional[float] = None
        self.baseline_throughput_tps: Optional[float] = None
        self.current_throughput_tps: Optional[float] = None
        self.attempted = 0
        self.promoted = 0
        self.rolled_back = 0
        self.events: List[OptimizationEvent] = []

    def set_baseline(self, latency_ms: float, throughput_tps: float) -> None:
        with self._lock:
            self.baseline_latency_ms = latency_ms
            self.baseline_throughput_tps = throughput_tps
            if self.current_latency_ms is None:
                self.current_latency_ms = latency_ms
            if self.current_throughput_tps is None:
                self.current_throughput_tps = throughput_tps

    def update_current(self, latency_ms: float, throughput_tps: float) -> None:
        with self._lock:
            self.current_latency_ms = latency_ms
            self.current_throughput_tps = throughput_tps

    def record_attempt(self, name: str, details: str = "") -> None:
        with self._lock:
            self.attempted += 1
            self.events.append(
                OptimizationEvent(name=name, action="attempted", timestamp=time.time(), details=details)
            )

    def record_promotion(self, name: str, details: str = "") -> None:
        with self._lock:
            self.promoted += 1
            self.events.append(
                OptimizationEvent(name=name, action="promoted", timestamp=time.time(), details=details)
            )

    def record_rollback(self, name: str, details: str = "") -> None:
        with self._lock:
            self.rolled_back += 1
            self.events.append(
                OptimizationEvent(name=name, action="rolled_back", timestamp=time.time(), details=details)
            )

    def _cost_improvement_pct_unlocked(self) -> Optional[float]:
        if (
            self.baseline_latency_ms is None
            or self.current_latency_ms is None
            or self.baseline_throughput_tps is None
            or self.current_throughput_tps is None
            or self.baseline_latency_ms == 0
            or self.baseline_throughput_tps == 0
        ):
            return None
        latency_gain = (
            (self.baseline_latency_ms - self.current_latency_ms) / self.baseline_latency_ms
        )
        throughput_gain = (
            (self.current_throughput_tps - self.baseline_throughput_tps)
            / self.baseline_throughput_tps
        )
        return round((latency_gain + throughput_gain) / 2 * 100, 2)

    def cost_improvement_pct(self) -> Optional[float]:
        with self._lock:
            return self._cost_improvement_pct_unlocked()

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            recent = [
                {
                    "name": event.name,
                    "action": event.action,
                    "timestamp": event.timestamp,
                    "details": event.details,
                }
                for event in self.events[-20:]
            ]
            return {
                "baseline_latency_ms": self.baseline_latency_ms,
                "current_latency_ms": self.current_latency_ms,
                "baseline_throughput_tps": self.baseline_throughput_tps,
                "current_throughput_tps": self.current_throughput_tps,
                "cost_improvement_pct": self._cost_improvement_pct_unlocked(),
                "optimizations_attempted": self.attempted,
                "optimizations_promoted": self.promoted,
                "optimizations_rolled_back": self.rolled_back,
                "recent_events": recent,
            }
