"""Synthetic workload generator for local inference benchmarking."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# A request fn returns (latency_ms, success) or (latency_ms, success, completion_tokens).
RequestFn = Callable[[str, int], tuple]


@dataclass(frozen=True)
class BenchProfile:
    name: str
    description: str
    prompts: List[str]


@dataclass
class BenchConfig:
    base_url: str = "http://127.0.0.1:8000"
    profile: str = "chat"
    duration_sec: float = 30.0
    concurrency: int = 2
    max_new_tokens: int = 16
    model: Optional[str] = None


@dataclass
class BenchResult:
    profile: str
    duration_sec: float
    concurrency: int
    requests_sent: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    client_avg_latency_ms: float = 0.0
    client_p95_latency_ms: float = 0.0
    client_tokens_per_sec: float = 0.0
    completion_tokens: int = 0
    wall_time_sec: float = 0.0
    server_tokens_per_sec: float = 0.0
    server_ttft_p95_ms: float = 0.0
    server_prefix_cache_hits: int = 0
    server_prefix_tokens_saved: int = 0
    latencies_ms: List[float] = field(default_factory=list)


RAG_SHARED_PREFIX = (
    "You are a helpful assistant. Context: "
    + ("KV blocks enable efficient attention reuse. " * 8)
)

PROFILES = {
    "chat": BenchProfile(
        name="chat",
        description="Short prompts with moderate concurrency",
        prompts=[
            "Explain KV cache in one sentence.",
            "What is continuous batching?",
            "Define prefill vs decode.",
            "How does prefix caching help RAG?",
            "Summarize paged attention.",
            "Why chunk long prefills?",
        ],
    ),
    "rag": BenchProfile(
        name="rag",
        description="Shared long prefix with short suffixes",
        prompts=[
            RAG_SHARED_PREFIX + " Question: Summarize the context.",
            RAG_SHARED_PREFIX + " Question: List three key ideas.",
            RAG_SHARED_PREFIX + " Question: What problem does paging solve?",
            RAG_SHARED_PREFIX + " Question: Why reuse prefixes?",
            RAG_SHARED_PREFIX + " Question: Give a one-line summary.",
            RAG_SHARED_PREFIX + " Question: What is the main topic?",
        ],
    ),
    "batch": BenchProfile(
        name="batch",
        description="Many unique prompts to reduce prefix reuse",
        prompts=[f"Unique benchmark prompt #{index}: describe token {index}." for index in range(1, 33)],
    ),
}


def list_profiles() -> dict[str, BenchProfile]:
    return dict(PROFILES)


def _percentile(values: List[float], p: float) -> float:
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


def default_request_fn(base_url: str, model: Optional[str] = None) -> RequestFn:
    endpoint = base_url.rstrip("/") + "/v1/completions"

    def send(prompt: str, max_new_tokens: int) -> tuple[float, bool, int]:
        # nifre reads `max_new_tokens`; vLLM/OpenAI read `max_tokens` and require
        # `model`. Sending all three keeps the same client fair to both servers.
        body: dict = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "max_tokens": max_new_tokens,
        }
        if model is not None:
            body["model"] = model
        payload = json.dumps(body).encode()
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
                ok = response.status == 200
            elapsed_ms = (time.perf_counter() - start) * 1000
            tokens = 0
            if ok:
                try:
                    usage = json.loads(body.decode()).get("usage", {})
                    tokens = int(usage.get("completion_tokens", 0))
                except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                    tokens = 0
            return elapsed_ms, ok, tokens
        except (urllib.error.URLError, TimeoutError):
            elapsed_ms = (time.perf_counter() - start) * 1000
            return elapsed_ms, False, 0

    return send


def fetch_server_metrics(base_url: str) -> dict:
    endpoint = base_url.rstrip("/") + "/observability/api/metrics"
    try:
        with urllib.request.urlopen(endpoint, timeout=5) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def run_bench(
    config: BenchConfig,
    *,
    request_fn: Optional[RequestFn] = None,
) -> BenchResult:
    profile = PROFILES.get(config.profile)
    if profile is None:
        raise ValueError(f"unknown profile {config.profile!r}; choose from {list(PROFILES)}")

    send = request_fn or default_request_fn(config.base_url, config.model)
    stop_at = time.time() + config.duration_sec
    lock = threading.Lock()
    prompt_index = 0
    latencies: List[float] = []
    sent = 0
    ok = 0
    failed = 0
    completion_tokens = 0

    def worker() -> None:
        nonlocal prompt_index, sent, ok, failed, completion_tokens
        while time.time() < stop_at:
            with lock:
                prompt = profile.prompts[prompt_index % len(profile.prompts)]
                prompt_index += 1
            result = send(prompt, config.max_new_tokens)
            latency_ms, success = result[0], result[1]
            tokens = result[2] if len(result) > 2 else 0
            with lock:
                sent += 1
                latencies.append(latency_ms)
                completion_tokens += tokens
                if success:
                    ok += 1
                else:
                    failed += 1

    wall_start = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(config.concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_time = time.perf_counter() - wall_start

    metrics = fetch_server_metrics(config.base_url)
    throughput = metrics.get("throughput", {})
    latency = metrics.get("latency", {}).get("ttft", {})

    return BenchResult(
        profile=config.profile,
        duration_sec=config.duration_sec,
        concurrency=config.concurrency,
        requests_sent=sent,
        requests_ok=ok,
        requests_failed=failed,
        client_avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        client_p95_latency_ms=_percentile(latencies, 95),
        client_tokens_per_sec=completion_tokens / wall_time if wall_time > 0 else 0.0,
        completion_tokens=completion_tokens,
        wall_time_sec=wall_time,
        server_tokens_per_sec=float(throughput.get("tokens_per_sec", 0.0)),
        server_ttft_p95_ms=float(latency.get("p95_ms", 0.0)),
        server_prefix_cache_hits=int(throughput.get("prefix_cache_hits", 0)),
        server_prefix_tokens_saved=int(throughput.get("prefix_cache_tokens_saved", 0)),
        latencies_ms=latencies,
    )


def format_report(result: BenchResult) -> str:
    lines = [
        f"Profile: {result.profile}",
        f"Duration: {result.duration_sec:.1f}s  Concurrency: {result.concurrency}",
        f"Requests: {result.requests_ok}/{result.requests_sent} ok ({result.requests_failed} failed)",
        f"Client latency avg/p95: {result.client_avg_latency_ms:.1f} ms / {result.client_p95_latency_ms:.1f} ms",
        f"Client tokens/sec: {result.client_tokens_per_sec:.2f}  ({result.completion_tokens} tok in {result.wall_time_sec:.1f}s)",
        f"Server tokens/sec: {result.server_tokens_per_sec:.2f}",
        f"Server TTFT p95: {result.server_ttft_p95_ms:.1f} ms",
        f"Prefix cache hits: {result.server_prefix_cache_hits}",
        f"Prefix tokens saved: {result.server_prefix_tokens_saved}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a running nifre inference server")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Inference server base URL")
    parser.add_argument(
        "--profile",
        default="chat",
        choices=sorted(PROFILES),
        help="Workload profile",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Benchmark duration in seconds")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent client threads")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Tokens to generate per request")
    parser.add_argument("--model", default=None, help="Model id to send in the request (required by vLLM)")
    args = parser.parse_args()

    config = BenchConfig(
        base_url=args.url,
        profile=args.profile,
        duration_sec=args.duration,
        concurrency=max(1, args.concurrency),
        max_new_tokens=max(1, args.max_new_tokens),
        model=args.model,
    )
    result = run_bench(config)
    print(format_report(result))


if __name__ == "__main__":
    main()
