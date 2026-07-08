"""Normalized workload metrics for the auto-tuning classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadSnapshot:
    """Point-in-time view of engine load derived from observability metrics."""

    timestamp: float
    queued_requests: int
    active_requests: int
    ttft_p95_ms: float
    ttft_p50_ms: float
    tokens_per_sec: float
    prefix_cache_hit_rate: float
    error_rate: float
    avg_prefill_tokens_per_step: float


def workload_snapshot_from_metrics(snapshot: dict) -> WorkloadSnapshot:
    """Build a :class:`WorkloadSnapshot` from ``MetricsCollector.snapshot()`` output."""
    health = snapshot.get("request_health", {})
    latency = snapshot.get("latency", {})
    throughput = snapshot.get("throughput", {})

    ttft = latency.get("ttft", {})
    tokens_saved = float(throughput.get("prefix_cache_tokens_saved", 0))
    total_prefill = float(throughput.get("total_prefill_tokens", 0))
    prefix_denominator = tokens_saved + total_prefill
    prefix_hit_rate = tokens_saved / prefix_denominator if prefix_denominator > 0 else 0.0

    return WorkloadSnapshot(
        timestamp=float(snapshot.get("timestamp", 0.0)),
        queued_requests=int(health.get("queued_requests", 0)),
        active_requests=int(health.get("active_requests", 0)),
        ttft_p95_ms=float(ttft.get("p95_ms", 0.0)),
        ttft_p50_ms=float(ttft.get("p50_ms", 0.0)),
        tokens_per_sec=float(throughput.get("tokens_per_sec", 0.0)),
        prefix_cache_hit_rate=prefix_hit_rate,
        error_rate=float(health.get("error_rate", 0.0)),
        avg_prefill_tokens_per_step=float(throughput.get("avg_prefill_tokens_per_step", 0.0)),
    )
