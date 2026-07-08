import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.backends.gpt import GptInferenceModel
from inference.batching import make_kv_cache
from inference.data_model import InferenceRequest
from inference.engine import Engine
from inference.model_runner import ModelRunner
from inference.models.gpt import GPT_CONFIG_124M, GptModel


def _tiny_gpt_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptInferenceModel(GptModel(cfg).to(device).eval())


def test_chunked_prefill_matches_single_step():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    runner = ModelRunner(model, device)
    cache = make_kv_cache(model.config, device)
    cache.init_batch(1)

    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    chunked = InferenceRequest(
        request_id="chunked",
        prompt_token_ids=prompt,
        max_new_tokens=2,
        prefill_chunk_size=3,
        batch_idx=0,
    )
    first_token_chunked = None
    while not chunked.prefill_complete:
        results = runner.prefill(cache, [chunked])
        if results[0] is not None:
            first_token_chunked = results[0]
            break

    cache.init_batch(1)
    full = InferenceRequest(
        request_id="full",
        prompt_token_ids=prompt,
        max_new_tokens=2,
        prefill_chunk_size=len(prompt),
        batch_idx=0,
    )
    first_token_full = runner.prefill(cache, [full])[0]

    assert chunked.prefill_offset == len(prompt)
    assert first_token_chunked == first_token_full


def test_engine_with_small_prefill_chunks():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(
        model,
        max_concurrent_requests=2,
        device=device,
        prefill_chunk_size=2,
    )

    request_id = engine.add_request([1, 2, 3, 4, 5], max_new_tokens=2)
    engine.run_until_done()

    result = engine.get_completed()[request_id]
    assert result.num_generated == 2
    assert result.prefill_offset == 5
    assert result.prefill_chunk_size == 2


def test_engine_add_request_uses_prefill_chunk_size():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    engine = Engine(
        model,
        max_concurrent_requests=2,
        device=device,
        prefill_chunk_size=64,
    )

    engine.add_request([1, 2, 3], max_new_tokens=1)
    assert engine.scheduler.waiting[0].prefill_chunk_size == 64
