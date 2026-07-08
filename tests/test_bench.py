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
