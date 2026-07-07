import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.gpt import GptInferenceModel
from inference.engine import Engine
from model.gpt_model import GPT_CONFIG_124M, GptModel


def _tiny_gpt_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptInferenceModel(GptModel(cfg).to(device).eval())


def test_engine_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)

    engine = Engine(model, max_concurrent_requests=2, device=device)
    engine.add_request([1, 2, 3], max_new_tokens=2)
    engine.add_request([10, 20, 30, 40], max_new_tokens=2)
    engine.add_request([5, 6], max_new_tokens=2)
    engine.run_until_done()

    assert len(engine.get_completed()) == 3
    for request in engine.get_completed().values():
        assert request.num_generated == 2


def test_token_callback_emits_all_generated_tokens():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(model, max_concurrent_requests=2, device=device)

    emitted: list[int] = []
    request_id = engine.add_request([1, 2, 3], max_new_tokens=3)
    engine.register_token_callback(request_id, emitted.append)

    while (
        request_id not in engine.scheduler.completed
        and request_id not in engine.scheduler.failed
    ):
        engine.step()

    engine.unregister_token_callback(request_id)

    result = engine.scheduler.completed[request_id]
    assert emitted == result.output_token_ids
    assert len(emitted) == 3


def test_stream_request_matches_generate():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(model, max_concurrent_requests=2, device=device)

    streamed = list(engine.stream_request([1, 2, 3], max_new_tokens=3))

    torch.manual_seed(0)
    engine2 = Engine(model, max_concurrent_requests=2, device=device)
    result = engine2.generate([1, 2, 3], max_new_tokens=3)

    assert streamed == result.output_token_ids
    assert len(streamed) == 3


def test_stream_request_with_chunked_prefill():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(
        model,
        max_concurrent_requests=2,
        device=device,
        prefill_chunk_size=2,
        max_tokens_per_step=4096,
    )

    streamed = list(engine.stream_request([1, 2, 3, 4, 5], max_new_tokens=2))

    assert len(streamed) == 2
    result = engine.scheduler.completed.values()
    assert len(result) == 1
    completed = next(iter(result))
    assert streamed == completed.output_token_ids
    assert completed.prefill_offset == 5


def test_stream_request_unregisters_callback():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(model, max_concurrent_requests=2, device=device)

    list(engine.stream_request([1, 2, 3], max_new_tokens=1))

    assert engine._token_callbacks == {}
