import json
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.backends.gpt import GptInferenceModel, TiktokenTokenizer
from inference.engine import Engine
from inference.server import create_app
from inference.models.gpt import GPT_CONFIG_124M, GptModel


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


def _parse_sse_events(lines):
    events = []
    done = False
    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            done = True
            break
        events.append(json.loads(data))
    return events, done


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


def test_completions_non_streaming_openai_shape():
    with _test_client() as client:
        for payload in (
            {"prompt": "hello", "max_new_tokens": 2},
            {"prompt": "hello", "max_new_tokens": 2, "stream": False},
        ):
            response = client.post("/v1/completions", json=payload)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/json")
            body = response.json()
            assert body["object"] == "text_completion"
            assert body["id"].startswith("cmpl-")
            assert body["model"] == "gpt"
            assert len(body["choices"]) == 1
            assert "text" in body["choices"][0]
            assert body["choices"][0]["finish_reason"] == "length"
            assert body["usage"]["prompt_tokens"] == 1
            assert body["usage"]["completion_tokens"] == 2


def test_completions_smoke():
    with _test_client() as client:
        response = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_new_tokens": 2},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "text_completion"
        assert payload["model"] == "gpt"
        assert payload["usage"]["completion_tokens"] == 2
        assert payload["choices"][0]["text"]


def test_completions_stream_openai_shape():
    with _test_client() as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "hello", "max_new_tokens": 3, "stream": True},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            events, done = _parse_sse_events(list(response.iter_lines()))

        assert done
        assert len(events) == 3

        completion_id = events[0]["id"]
        created = events[0]["created"]
        for i, event in enumerate(events):
            assert event["id"] == completion_id
            assert event["created"] == created
            assert event["object"] == "text_completion"
            assert event["model"] == "gpt"
            assert len(event["choices"]) == 1
            assert event["choices"][0]["index"] == 0
            assert "text" in event["choices"][0]
            if i < len(events) - 1:
                assert event["choices"][0]["finish_reason"] is None
            else:
                assert event["choices"][0]["finish_reason"] == "length"

        blocking = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_new_tokens": 3, "stream": False},
        ).json()
        streamed_text = "".join(event["choices"][0]["text"] for event in events)
        assert streamed_text == blocking["choices"][0]["text"]


def test_completions_stream_respects_requested_model_name():
    with _test_client() as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "prompt": "hello",
                "max_new_tokens": 1,
                "stream": True,
                "model": "my-custom-model",
            },
        ) as response:
            events, done = _parse_sse_events(list(response.iter_lines()))

        assert done
        assert events[0]["model"] == "my-custom-model"
