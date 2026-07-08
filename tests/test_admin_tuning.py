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


def _test_client(*, auto_tune: bool = False):
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

    app = create_app(
        engine=engine,
        tokenizer=tokenizer,
        backend_name="gpt",
        auto_tune=auto_tune,
    )
    return TestClient(app)


def test_admin_tuning_get_status():
    with _test_client() as client:
        response = client.get("/v1/admin/tuning")
        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is True
        assert payload["enabled"] is False
        assert payload["goal"] == "balanced"
        assert payload["engine_config"]["prefill_chunk_size"] == 128


def test_admin_tuning_enable_and_update_goal():
    with _test_client() as client:
        enabled = client.post("/v1/admin/tuning", json={"enabled": True})
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True

        updated = client.post("/v1/admin/tuning", json={"goal": "latency"})
        assert updated.status_code == 200
        assert updated.json()["goal"] == "latency"

        disabled = client.post("/v1/admin/tuning", json={"enabled": False})
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False


def test_observability_tuning_route():
    with _test_client() as client:
        response = client.get("/observability/api/tuning")
        assert response.status_code == 200
        assert response.json()["available"] is True


def test_health_exposes_auto_tune_endpoint():
    with _test_client() as client:
        payload = client.get("/health").json()
        assert payload["auto_tune"] == "/v1/admin/tuning"


def test_auto_tune_flag_starts_enabled():
    with _test_client(auto_tune=True) as client:
        payload = client.get("/v1/admin/tuning").json()
        assert payload["enabled"] is True
