import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.gpt import GptInferenceModel, TiktokenTokenizer
from inference.engine import Engine
from inference.server import create_app
from model.gpt_model import GPT_CONFIG_124M, GptModel
from observability import Observability
from observability.metrics_store import latency_summary, percentile
from observability.optimization import OptimizationTracker


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert latency_summary(values)["p50_ms"] == 3000.0


def test_engine_metrics_smoke():
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64

    device = torch.device("cpu")
    model = GptModel(cfg).to(device).eval()
    wrapped = GptInferenceModel(model)
    obs = Observability(model_name="gpt-test", runtime="custom", precision="fp16")
    engine = Engine(
        wrapped,
        max_concurrent_requests=2,
        device=device,
        metrics_collector=obs.collector,
        prefill_chunk_size=2,
    )
    obs.attach(engine)

    engine.generate([1, 2, 3, 4, 5], max_new_tokens=2)

    snapshot = obs.snapshot()
    assert snapshot["request_health"]["completed_requests"] == 1
    assert snapshot["throughput"]["tokens_per_request"] > 0
    assert snapshot["throughput"]["total_prefill_tokens"] == 5
    assert snapshot["throughput"]["avg_prefill_tokens_per_step"] > 0
    assert snapshot["gpu_runtime"]["model_name"] == "gpt-test"
    engine_config = snapshot["gpu_runtime"]["engine_config"]
    assert engine_config["cache_type"] == "paged"
    assert engine_config["use_paged_kv_cache"] is True
    assert engine_config["prefill_chunk_size"] == 2
    assert engine_config["max_tokens_per_step"] == 2048
    assert engine_config["block_size"] == wrapped.config.block_size
    assert snapshot["latency"]["ttft"]["p50_ms"] >= 0
    assert snapshot["optimization_history"]["baseline_latency_ms"] is not None

    promoted = [e["name"] for e in snapshot["optimization_history"]["recent_events"] if e["action"] == "promoted"]
    assert "paged-kv-cache" in promoted
    assert "chunked-prefill" in promoted
    assert "prefix-cache" in promoted


def test_optimization_tracker():
    tracker = OptimizationTracker()
    tracker.set_baseline(100.0, 50.0)
    tracker.record_attempt("flash-attn", details="trial run")
    tracker.record_promotion("flash-attn")
    tracker.update_current(80.0, 65.0)
    snap = tracker.snapshot()
    assert snap["optimizations_attempted"] == 1
    assert snap["optimizations_promoted"] == 1
    assert snap["cost_improvement_pct"] is not None


def test_observability_routes():
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64

    device = torch.device("cpu")
    model = GptModel(cfg).to(device).eval()
    wrapped = GptInferenceModel(model)
    engine = Engine(wrapped, max_concurrent_requests=2, device=device)
    tokenizer = TiktokenTokenizer(pad_token_id=wrapped.config.pad_token_id)

    app = create_app(engine=engine, tokenizer=tokenizer, backend_name="gpt")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json()["observability"] == "/observability"

        metrics = client.get("/observability/api/metrics")
        assert metrics.status_code == 200
        assert "request_health" in metrics.json()

        dashboard = client.get("/observability")
        assert dashboard.status_code == 200
        assert "Inference Engine Observability" in dashboard.text

        client.post("/v1/completions", json={"prompt": "hi", "max_new_tokens": 1})
        after = client.get("/observability/api/metrics").json()
        assert after["request_health"]["completed_requests"] >= 1
