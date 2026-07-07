import sys
import threading
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.gpt import GptInferenceModel
from inference.engine import Engine
from inference.engine_worker import EngineWorker
from model.gpt_model import GPT_CONFIG_124M, GptModel


def _tiny_gpt_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptInferenceModel(GptModel(cfg).to(device).eval())


def _worker(device=None) -> EngineWorker:
    device = device or torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(model, max_concurrent_requests=2, device=device)
    worker = EngineWorker(engine)
    worker.start()
    return worker


def test_worker_generate_matches_engine():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)

    worker = EngineWorker(Engine(model, max_concurrent_requests=2, device=device))
    worker.start()
    try:
        result = worker.generate([1, 2, 3], max_new_tokens=3)
    finally:
        worker.stop()

    torch.manual_seed(0)
    engine = Engine(model, max_concurrent_requests=2, device=device)
    expected = engine.generate([1, 2, 3], max_new_tokens=3)

    assert result.output_token_ids == expected.output_token_ids
    assert len(result.output_token_ids) == 3


def test_worker_generate_stream_matches_generate():
    torch.manual_seed(0)
    worker = _worker()
    try:
        streamed = list(worker.generate_stream([1, 2, 3], max_new_tokens=3))
        result = worker.generate([1, 2, 3], max_new_tokens=3)
    finally:
        worker.stop()

    torch.manual_seed(0)
    worker2 = _worker()
    try:
        expected = worker2.generate([1, 2, 3], max_new_tokens=3)
    finally:
        worker2.stop()

    assert streamed == expected.output_token_ids


def test_worker_concurrent_generates():
    torch.manual_seed(0)
    worker = _worker()
    results: dict[str, object] = {}
    errors: list[Exception] = []

    def run_generate(name: str, prompt: list[int]):
        try:
            results[name] = worker.generate(prompt, max_new_tokens=2)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run_generate, args=("a", [1, 2, 3])),
        threading.Thread(target=run_generate, args=("b", [10, 20, 30, 40])),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    try:
        worker.stop()
    finally:
        pass

    assert not errors
    assert len(results) == 2
    assert results["a"].num_generated == 2
    assert results["b"].num_generated == 2


def test_worker_concurrent_stream_and_generate():
    torch.manual_seed(0)
    worker = _worker()
    streamed: list[int] = []
    blocking = None
    errors: list[Exception] = []

    def run_stream():
        nonlocal streamed
        try:
            streamed = list(worker.generate_stream([1, 2, 3], max_new_tokens=2))
        except Exception as exc:
            errors.append(exc)

    def run_generate():
        nonlocal blocking
        try:
            blocking = worker.generate([10, 20, 30, 40], max_new_tokens=2)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run_stream),
        threading.Thread(target=run_generate),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    worker.stop()

    assert not errors
    assert len(streamed) == 2
    assert blocking is not None
    assert blocking.num_generated == 2
