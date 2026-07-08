"""Fetch and normalize server-side metrics from nifre or vLLM."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerMetrics:
    """Normalized metrics for side-by-side comparison."""

    source: str = "unknown"  # "nifre" | "vllm" | "unknown"
    tokens_per_sec: float | None = None
    output_tokens_per_sec: float | None = None
    input_tokens_per_sec: float | None = None
    requests_per_sec: float | None = None
    requests_completed: int | None = None
    error_rate: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    total_latency_p50_ms: float | None = None
    total_latency_p95_ms: float | None = None
    decode_step_p95_ms: float | None = None
    inter_token_p95_ms: float | None = None
    gpu_utilization_pct: float | None = None
    gpu_memory_gb: float | None = None
    kv_cache_utilization_pct: float | None = None
    prefix_cache_hits: int | None = None
    prefix_tokens_saved: int | None = None
    prefix_hit_rate: float | None = None
    prefix_cache_entries: int | None = None
    prefix_cache_memory_mb: float | None = None
    prefix_cache_reuse_ratio: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _get(url: str, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _lat_ms(latency_block: dict | None, key: str = "p95_ms") -> float | None:
    if not latency_block:
        return None
    value = latency_block.get(key)
    return float(value) if value is not None else None


def from_nifre_observability(payload: dict) -> ServerMetrics:
    throughput = payload.get("throughput", {})
    health = payload.get("request_health", {})
    latency = payload.get("latency", {})
    runtime = payload.get("gpu_runtime", {})
    prefix = (runtime.get("engine_config") or {}).get("prefix_cache") or {}

    return ServerMetrics(
        source="nifre",
        tokens_per_sec=_float(throughput.get("tokens_per_sec")),
        output_tokens_per_sec=_float(throughput.get("output_tokens_per_sec")),
        input_tokens_per_sec=_float(throughput.get("input_tokens_per_sec")),
        requests_per_sec=_float(health.get("requests_per_sec")),
        requests_completed=_int(health.get("completed_requests")),
        error_rate=_float(health.get("error_rate")),
        ttft_p50_ms=_lat_ms(latency.get("ttft"), "p50_ms"),
        ttft_p95_ms=_lat_ms(latency.get("ttft"), "p95_ms"),
        total_latency_p50_ms=_lat_ms(latency.get("total_request_latency"), "p50_ms"),
        total_latency_p95_ms=_lat_ms(latency.get("total_request_latency"), "p95_ms"),
        decode_step_p95_ms=_lat_ms(latency.get("decode_step_latency"), "p95_ms"),
        inter_token_p95_ms=_lat_ms(latency.get("inter_token_latency"), "p95_ms"),
        gpu_utilization_pct=_float(runtime.get("gpu_utilization_pct")),
        gpu_memory_gb=_float(runtime.get("gpu_memory_used_gb")),
        kv_cache_utilization_pct=_float(runtime.get("kv_cache_utilization_pct")),
        prefix_cache_hits=_int(throughput.get("prefix_cache_hits")),
        prefix_tokens_saved=_int(throughput.get("prefix_cache_tokens_saved")),
        prefix_hit_rate=_float(prefix.get("hit_rate")),
        prefix_cache_entries=_int(prefix.get("entries")),
        prefix_cache_memory_mb=_float(prefix.get("memory_mb")),
        prefix_cache_reuse_ratio=_float(throughput.get("prefix_cache_reuse_ratio")),
    )


_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$"
)


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse simple Prometheus text exposition (gauges/counters, no labels aggregation)."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if match:
            values[match.group("name")] = float(match.group("value"))
    return values


def _find_metric(values: dict[str, float], *candidates: str) -> float | None:
    for name in candidates:
        if name in values:
            return values[name]
    for key, value in values.items():
        for candidate in candidates:
            if key.endswith(candidate) or candidate in key:
                return value
    return None


def from_vllm_prometheus(text: str) -> ServerMetrics:
    values = _parse_prometheus(text)

    prompt_tps = _find_metric(
        values,
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm_avg_prompt_throughput_toks_per_s",
    )
    gen_tps = _find_metric(
        values,
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm_avg_generation_throughput_toks_per_s",
    )
    total_tps = None
    if prompt_tps is not None and gen_tps is not None:
        total_tps = prompt_tps + gen_tps

    gpu_cache = _find_metric(
        values,
        "vllm:gpu_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    )
    prefix_hits = _find_metric(
        values,
        "vllm:prefix_cache_hits_total",
        "vllm_prefix_cache_hits_total",
    )
    prefix_queries = _find_metric(
        values,
        "vllm:prefix_cache_queries_total",
        "vllm_prefix_cache_queries_total",
    )
    prefix_hit_rate = None
    if prefix_hits is not None and prefix_queries and prefix_queries > 0:
        prefix_hit_rate = prefix_hits / prefix_queries

    running = _find_metric(values, "vllm:num_requests_running", "vllm_num_requests_running")
    waiting = _find_metric(values, "vllm:num_requests_waiting", "vllm_num_requests_waiting")

    return ServerMetrics(
        source="vllm",
        tokens_per_sec=total_tps,
        output_tokens_per_sec=gen_tps,
        input_tokens_per_sec=prompt_tps,
        kv_cache_utilization_pct=gpu_cache,
        prefix_cache_hits=int(prefix_hits) if prefix_hits is not None else None,
        prefix_hit_rate=prefix_hit_rate,
        extra={
            "requests_running": int(running) if running is not None else None,
            "requests_waiting": int(waiting) if waiting is not None else None,
        },
    )


def fetch_server_metrics(base_url: str) -> ServerMetrics:
    """Try nifre observability JSON, then vLLM Prometheus ``/metrics``."""
    root = base_url.rstrip("/")

    obs_text = _get(f"{root}/observability/api/metrics")
    if obs_text:
        try:
            payload = json.loads(obs_text)
            if "throughput" in payload or "latency" in payload:
                return from_nifre_observability(payload)
        except json.JSONDecodeError:
            pass

    prom_text = _get(f"{root}/metrics")
    if prom_text and ("vllm" in prom_text.lower() or "# TYPE" in prom_text):
        return from_vllm_prometheus(prom_text)

    return ServerMetrics(source="unknown")


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
