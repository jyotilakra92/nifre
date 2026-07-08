"""Side-by-side A/B benchmark of two OpenAI-compatible completion servers.

Engine-agnostic: it only uses client-measured latency and throughput (from each
response's ``usage.completion_tokens``), so it compares nifre against vLLM (or
any ``/v1/completions`` server) fairly on the same prompts and load.

Example (on a CUDA box, same weights on both):

    # nifre on :8000, vLLM on :8001
    PYTHONPATH=src python3 -m compare \
        --a-url http://127.0.0.1:8000 --a-label nifre \
        --b-url http://127.0.0.1:8001 --b-label vllm \
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


def _warmup(base_url: str, count: int, max_new_tokens: int) -> None:
    send = default_request_fn(base_url)
    for _ in range(count):
        send("Warmup request, respond briefly.", max_new_tokens)


def _bench(base_url: str, config: BenchConfig) -> BenchResult:
    return run_bench(BenchConfig(
        base_url=base_url,
        profile=config.profile,
        duration_sec=config.duration_sec,
        concurrency=config.concurrency,
        max_new_tokens=config.max_new_tokens,
    ))


def _ratio(a: float, b: float) -> str:
    if b == 0:
        return "n/a"
    return f"{a / b:.2f}x"


def format_comparison(
    a_label: str,
    b_label: str,
    a: BenchResult,
    b: BenchResult,
) -> str:
    rows = [
        ("requests ok", f"{a.requests_ok}/{a.requests_sent}", f"{b.requests_ok}/{b.requests_sent}", ""),
        ("client tokens/sec", f"{a.client_tokens_per_sec:.1f}", f"{b.client_tokens_per_sec:.1f}",
         _ratio(a.client_tokens_per_sec, b.client_tokens_per_sec)),
        ("avg latency ms", f"{a.client_avg_latency_ms:.1f}", f"{b.client_avg_latency_ms:.1f}",
         _ratio(b.client_avg_latency_ms, a.client_avg_latency_ms)),
        ("p95 latency ms", f"{a.client_p95_latency_ms:.1f}", f"{b.client_p95_latency_ms:.1f}",
         _ratio(b.client_p95_latency_ms, a.client_p95_latency_ms)),
    ]

    metric_w = max(len(r[0]) for r in rows)
    a_w = max(len(a_label), max(len(r[1]) for r in rows))
    b_w = max(len(b_label), max(len(r[2]) for r in rows))

    header = f"{'metric':<{metric_w}}  {a_label:>{a_w}}  {b_label:>{b_w}}  {'A/B':>6}"
    lines = [header, "-" * len(header)]
    for name, av, bv, ratio in rows:
        lines.append(f"{name:<{metric_w}}  {av:>{a_w}}  {bv:>{b_w}}  {ratio:>6}")
    lines.append("")
    lines.append("(A/B > 1.0 favors A: higher tokens/sec, lower latency)")
    return "\n".join(lines)


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
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests per server")
    args = parser.parse_args()

    config = BenchConfig(
        profile=args.profile,
        duration_sec=args.duration,
        concurrency=max(1, args.concurrency),
        max_new_tokens=max(1, args.max_new_tokens),
    )

    print(f"Warming up {args.a_label} and {args.b_label} ({args.warmup} requests each)...")
    _warmup(args.a_url, args.warmup, config.max_new_tokens)
    _warmup(args.b_url, args.warmup, config.max_new_tokens)

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
