"""Side-by-side A/B benchmark of two OpenAI-compatible completion servers.

Compares client-measured latency/throughput plus server-side metrics from
nifre's ``/observability/api/metrics`` JSON and vLLM's ``/metrics`` Prometheus
endpoint (throughput, TTFT, GPU/KV utilization, prefix cache, etc.).

Example (on a CUDA box, same weights on both):

    PYTHONPATH=src python3 -m compare \
        --a-url http://127.0.0.1:8000 --a-label nifre \
        --b-url http://127.0.0.1:8001 --b-label vllm \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --profile rag --duration 60 --concurrency 8 --max-new-tokens 64
"""

from __future__ import annotations

import argparse

from bench import (
    BenchConfig,
    BenchResult,
    PROFILES,
    default_request_fn,
    run_bench,
)
from server_metrics import ServerMetrics


def _warmup(base_url: str, count: int, max_new_tokens: int, model: str | None) -> None:
    send = default_request_fn(base_url, model)
    for _ in range(count):
        send("Warmup request, respond briefly.", max_new_tokens)


def _bench(base_url: str, config: BenchConfig) -> BenchResult:
    return run_bench(BenchConfig(
        base_url=base_url,
        profile=config.profile,
        duration_sec=config.duration_sec,
        concurrency=config.concurrency,
        max_new_tokens=config.max_new_tokens,
        model=config.model,
    ))


def _ratio(a: float | None, b: float | None, *, higher_is_better: bool = True) -> str:
    if a is None or b is None or b == 0:
        return "n/a"
    ratio = a / b
    if not higher_is_better:
        ratio = b / a if a else 0
    return f"{ratio:.2f}x"


def _fmt(value: float | int | None, *, suffix: str = "", precision: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{precision}f}{suffix}"


def _metrics(result: BenchResult) -> ServerMetrics:
    return result.server_metrics or ServerMetrics(source="unknown")


def _comparison_table(
    a_label: str,
    b_label: str,
    rows: list[tuple[str, str, str, str]],
) -> list[str]:
    if not rows:
        return []
    metric_w = max(len(r[0]) for r in rows)
    a_w = max(len(a_label), max(len(r[1]) for r in rows))
    b_w = max(len(b_label), max(len(r[2]) for r in rows))
    header = f"{'metric':<{metric_w}}  {a_label:>{a_w}}  {b_label:>{b_w}}  {'A/B':>6}"
    lines = [header, "-" * len(header)]
    for name, av, bv, ratio in rows:
        lines.append(f"{name:<{metric_w}}  {av:>{a_w}}  {bv:>{b_w}}  {ratio:>6}")
    return lines


def format_comparison(a_label: str, b_label: str, a: BenchResult, b: BenchResult) -> str:
    ma, mb = _metrics(a), _metrics(b)
    sections: list[str] = []

    client_rows = [
        ("requests ok", f"{a.requests_ok}/{a.requests_sent}", f"{b.requests_ok}/{b.requests_sent}", ""),
        ("client tokens/sec", _fmt(a.client_tokens_per_sec), _fmt(b.client_tokens_per_sec),
         _ratio(a.client_tokens_per_sec, b.client_tokens_per_sec)),
        ("client avg latency ms", _fmt(a.client_avg_latency_ms), _fmt(b.client_avg_latency_ms),
         _ratio(a.client_avg_latency_ms, b.client_avg_latency_ms, higher_is_better=False)),
        ("client p95 latency ms", _fmt(a.client_p95_latency_ms), _fmt(b.client_p95_latency_ms),
         _ratio(a.client_p95_latency_ms, b.client_p95_latency_ms, higher_is_better=False)),
        ("completion tokens", str(a.completion_tokens), str(b.completion_tokens), ""),
    ]
    sections.append("=== Client (measured by harness) ===")
    sections.extend(_comparison_table(a_label, b_label, client_rows))

    throughput_rows = [
        ("server tokens/sec", _fmt(ma.tokens_per_sec), _fmt(mb.tokens_per_sec),
         _ratio(ma.tokens_per_sec, mb.tokens_per_sec)),
        ("output tokens/sec", _fmt(ma.output_tokens_per_sec), _fmt(mb.output_tokens_per_sec),
         _ratio(ma.output_tokens_per_sec, mb.output_tokens_per_sec)),
        ("input tokens/sec", _fmt(ma.input_tokens_per_sec), _fmt(mb.input_tokens_per_sec),
         _ratio(ma.input_tokens_per_sec, mb.input_tokens_per_sec)),
        ("requests/sec", _fmt(ma.requests_per_sec, precision=2), _fmt(mb.requests_per_sec, precision=2),
         _ratio(ma.requests_per_sec, mb.requests_per_sec)),
        ("requests completed", _fmt(ma.requests_completed), _fmt(mb.requests_completed), ""),
        ("error rate", _fmt(ma.error_rate, precision=4), _fmt(mb.error_rate, precision=4), ""),
    ]
    sections.append("")
    sections.append("=== Throughput (server-reported) ===")
    sections.extend(_comparison_table(a_label, b_label, throughput_rows))

    latency_rows = [
        ("TTFT p50 ms", _fmt(ma.ttft_p50_ms), _fmt(mb.ttft_p50_ms),
         _ratio(ma.ttft_p50_ms, mb.ttft_p50_ms, higher_is_better=False)),
        ("TTFT p95 ms", _fmt(ma.ttft_p95_ms), _fmt(mb.ttft_p95_ms),
         _ratio(ma.ttft_p95_ms, mb.ttft_p95_ms, higher_is_better=False)),
        ("total latency p50 ms", _fmt(ma.total_latency_p50_ms), _fmt(mb.total_latency_p50_ms),
         _ratio(ma.total_latency_p50_ms, mb.total_latency_p50_ms, higher_is_better=False)),
        ("total latency p95 ms", _fmt(ma.total_latency_p95_ms), _fmt(mb.total_latency_p95_ms),
         _ratio(ma.total_latency_p95_ms, mb.total_latency_p95_ms, higher_is_better=False)),
        ("decode step p95 ms", _fmt(ma.decode_step_p95_ms), _fmt(mb.decode_step_p95_ms),
         _ratio(ma.decode_step_p95_ms, mb.decode_step_p95_ms, higher_is_better=False)),
        ("inter-token p95 ms", _fmt(ma.inter_token_p95_ms), _fmt(mb.inter_token_p95_ms),
         _ratio(ma.inter_token_p95_ms, mb.inter_token_p95_ms, higher_is_better=False)),
    ]
    sections.append("")
    sections.append("=== Latency (server-reported) ===")
    sections.extend(_comparison_table(a_label, b_label, latency_rows))

    resource_rows = [
        ("GPU util %", _fmt(ma.gpu_utilization_pct), _fmt(mb.gpu_utilization_pct),
         _ratio(ma.gpu_utilization_pct, mb.gpu_utilization_pct)),
        ("GPU memory GB", _fmt(ma.gpu_memory_gb, precision=2), _fmt(mb.gpu_memory_gb, precision=2), ""),
        ("KV cache util %", _fmt(ma.kv_cache_utilization_pct), _fmt(mb.kv_cache_utilization_pct),
         _ratio(ma.kv_cache_utilization_pct, mb.kv_cache_utilization_pct)),
    ]
    if mb.extra.get("requests_running") is not None:
        resource_rows.append((
            "requests running/waiting",
            "n/a",
            f"{mb.extra.get('requests_running', 'n/a')}/{mb.extra.get('requests_waiting', 'n/a')}",
            "",
        ))
    sections.append("")
    sections.append("=== GPU / KV cache ===")
    sections.extend(_comparison_table(a_label, b_label, resource_rows))

    prefix_rows = [
        ("prefix cache hits", _fmt(ma.prefix_cache_hits), _fmt(mb.prefix_cache_hits),
         _ratio(ma.prefix_cache_hits, mb.prefix_cache_hits)),
        ("prefix tokens saved", _fmt(ma.prefix_tokens_saved), _fmt(mb.prefix_tokens_saved),
         _ratio(ma.prefix_tokens_saved, mb.prefix_tokens_saved)),
        ("prefix hit rate", _fmt(ma.prefix_hit_rate, precision=4), _fmt(mb.prefix_hit_rate, precision=4),
         _ratio(ma.prefix_hit_rate, mb.prefix_hit_rate)),
        ("prefix reuse ratio", _fmt(ma.prefix_cache_reuse_ratio, precision=4),
         _fmt(mb.prefix_cache_reuse_ratio, precision=4),
         _ratio(ma.prefix_cache_reuse_ratio, mb.prefix_cache_reuse_ratio)),
        ("prefix entries", _fmt(ma.prefix_cache_entries), _fmt(mb.prefix_cache_entries), ""),
        ("prefix cache MB", _fmt(ma.prefix_cache_memory_mb, precision=2),
         _fmt(mb.prefix_cache_memory_mb, precision=2), ""),
    ]
    sections.append("")
    sections.append("=== Prefix cache ===")
    sections.extend(_comparison_table(a_label, b_label, prefix_rows))

    sections.append("")
    sections.append(f"Metric sources: {a_label}={ma.source}, {b_label}={mb.source}")
    sections.append("(A/B > 1.0 favors A for throughput/utilization; latency rows invert so >1.0 still favors A)")
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark two completion servers")
    parser.add_argument("--a-url", default="http://127.0.0.1:8000")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-url", default="http://127.0.0.1:8001")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--profile", default="rag", choices=sorted(PROFILES))
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model", default=None, help="Model id sent in each request (required by vLLM)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests per server")
    args = parser.parse_args()

    config = BenchConfig(
        profile=args.profile,
        duration_sec=args.duration,
        concurrency=max(1, args.concurrency),
        max_new_tokens=max(1, args.max_new_tokens),
        model=args.model,
    )

    print(f"Warming up {args.a_label} and {args.b_label} ({args.warmup} requests each)...")
    _warmup(args.a_url, args.warmup, config.max_new_tokens, config.model)
    _warmup(args.b_url, args.warmup, config.max_new_tokens, config.model)

    print(f"Benchmarking {args.a_label} ({args.a_url})...")
    result_a = _bench(args.a_url, config)
    print(f"Benchmarking {args.b_label} ({args.b_url})...")
    result_b = _bench(args.b_url, config)

    print()
    print(f"Profile: {config.profile}  Duration: {config.duration_sec:.0f}s  "
          f"Concurrency: {config.concurrency}  Max new tokens: {config.max_new_tokens}")
    print()
    print(format_comparison(args.a_label, args.b_label, result_a, result_b))


if __name__ == "__main__":
    main()
