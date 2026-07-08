import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from server_metrics import from_nifre_observability, from_vllm_prometheus


def test_from_nifre_observability_extracts_key_fields():
    payload = {
        "throughput": {
            "tokens_per_sec": 250.0,
            "output_tokens_per_sec": 80.0,
            "input_tokens_per_sec": 170.0,
            "prefix_cache_hits": 100,
            "prefix_cache_tokens_saved": 1600,
            "prefix_cache_reuse_ratio": 0.45,
        },
        "request_health": {
            "requests_per_sec": 2.5,
            "completed_requests": 150,
            "error_rate": 0.0,
        },
        "latency": {
            "ttft": {"p50_ms": 70.0, "p95_ms": 90.0},
            "total_request_latency": {"p50_ms": 2000.0, "p95_ms": 2500.0},
            "decode_step_latency": {"p95_ms": 65.0},
            "inter_token_latency": {"p95_ms": 72.0},
        },
        "gpu_runtime": {
            "gpu_utilization_pct": 85.0,
            "gpu_memory_used_gb": 4.2,
            "kv_cache_utilization_pct": 17.0,
            "engine_config": {
                "prefix_cache": {
                    "hit_rate": 0.97,
                    "entries": 4,
                    "memory_mb": 0.75,
                }
            },
        },
    }
    metrics = from_nifre_observability(payload)
    assert metrics.source == "nifre"
    assert metrics.tokens_per_sec == 250.0
    assert metrics.ttft_p95_ms == 90.0
    assert metrics.prefix_cache_hits == 100
    assert metrics.prefix_hit_rate == 0.97
    assert metrics.prefix_cache_entries == 4


def test_from_vllm_prometheus_extracts_throughput_and_cache():
    text = """
# HELP vllm:avg_prompt_throughput_toks_per_s Avg prefill throughput
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s 120.5
vllm:avg_generation_throughput_toks_per_s 45.2
vllm:gpu_cache_usage_perc 33.1
vllm:prefix_cache_hits_total 88
vllm:prefix_cache_queries_total 90
vllm:num_requests_running 4
vllm:num_requests_waiting 0
"""
    metrics = from_vllm_prometheus(text)
    assert metrics.source == "vllm"
    assert metrics.input_tokens_per_sec == 120.5
    assert metrics.output_tokens_per_sec == 45.2
    assert metrics.tokens_per_sec == 165.7
    assert metrics.kv_cache_utilization_pct == 33.1
    assert metrics.prefix_cache_hits == 88
    assert abs(metrics.prefix_hit_rate - 88 / 90) < 1e-6
    assert metrics.extra["requests_running"] == 4
