import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bench import BenchConfig, PROFILES, format_report, list_profiles, run_bench


def test_profiles_exist():
    profiles = list_profiles()
    assert set(profiles) == {"chat", "rag", "batch"}
    assert len(profiles["rag"].prompts) >= 4
    shared = profiles["rag"].prompts[0][:120]
    assert all(prompt.startswith(shared) for prompt in profiles["rag"].prompts)


def test_run_bench_with_mock_request_fn():
    calls = {"count": 0}

    def request_fn(prompt: str, max_new_tokens: int):
        calls["count"] += 1
        return 12.5, True

    config = BenchConfig(profile="chat", duration_sec=0.2, concurrency=2, max_new_tokens=4)
    result = run_bench(config, request_fn=request_fn)

    assert result.requests_sent >= 1
    assert result.requests_ok == result.requests_sent
    assert result.requests_failed == 0
    assert result.client_avg_latency_ms == 12.5
    assert calls["count"] == result.requests_sent


def test_run_bench_counts_completion_tokens():
    def request_fn(prompt: str, max_new_tokens: int):
        return 10.0, True, 7

    config = BenchConfig(profile="chat", duration_sec=0.2, concurrency=2, max_new_tokens=4)
    result = run_bench(config, request_fn=request_fn)

    assert result.completion_tokens == result.requests_sent * 7
    assert result.wall_time_sec > 0
    assert result.client_tokens_per_sec > 0


def test_compare_format_reports_both_labels_and_ratio():
    from bench import BenchResult
    from compare import format_comparison

    a = BenchResult(profile="rag", duration_sec=1.0, concurrency=2, requests_ok=10,
                    requests_sent=10, client_tokens_per_sec=200.0, client_avg_latency_ms=100.0,
                    client_p95_latency_ms=150.0)
    b = BenchResult(profile="rag", duration_sec=1.0, concurrency=2, requests_ok=10,
                    requests_sent=10, client_tokens_per_sec=400.0, client_avg_latency_ms=50.0,
                    client_p95_latency_ms=80.0)
    text = format_comparison("nifre", "vllm", a, b)
    assert "nifre" in text and "vllm" in text
    assert "client tokens/sec" in text
    assert "Prefix cache" in text
    assert "TTFT p95 ms" in text
    assert "0.50x" in text  # nifre 200 / vllm 400


def test_run_bench_unknown_profile():
    try:
        run_bench(BenchConfig(profile="missing"))
        raise AssertionError("expected ValueError for unknown profile")
    except ValueError as exc:
        assert "unknown profile" in str(exc)


def test_format_report():
    from bench import BenchResult

    text = format_report(
        BenchResult(
            profile="chat",
            duration_sec=1.0,
            concurrency=2,
            requests_sent=4,
            requests_ok=4,
        )
    )
    assert "Profile: chat" in text
    assert "4/4 ok" in text
