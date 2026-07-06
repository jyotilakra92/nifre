import sys
from pathlib import Path

import tiktoken
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


def _test_client():
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
    )
    return TestClient(app)


def test_health():
    with _test_client() as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["model"] == "gpt"
        assert "gpt" in payload["backends"]


def test_completions_validation():
    with _test_client() as client:
        response = client.post("/v1/completions", json={"max_new_tokens": 5})
        assert response.status_code == 422


def test_completions_smoke():
    with _test_client() as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_new_tokens": 2},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["prompt"] == "hello"
        assert payload["model"] == "gpt"
        assert len(payload["output_token_ids"]) == 2
        assert payload["text"]
