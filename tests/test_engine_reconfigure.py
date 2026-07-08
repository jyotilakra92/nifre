import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.backends.gpt import GptInferenceModel
from inference.data_model import EngineConfig, InferenceRequest
from inference.engine import Engine
from inference.scheduler import Scheduler
from inference.models.gpt import GPT_CONFIG_124M, GptModel


def _tiny_gpt_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptInferenceModel(GptModel(cfg).to(device).eval())


def _engine(**kwargs):
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)
    defaults = dict(max_concurrent_requests=2, device=device)
    defaults.update(kwargs)
    return Engine(model, **defaults)


def test_get_config_returns_current_settings():
    engine = _engine(prefill_chunk_size=64, max_tokens_per_step=512)
    config = engine.get_config()

    assert config == EngineConfig(
        max_concurrent_requests=2,
        prefill_chunk_size=64,
        max_tokens_per_step=512,
        use_paged_kv_cache=True,
        use_prefix_cache=True,
    )


def test_reconfigure_round_trip():
    engine = _engine()
    updated = engine.reconfigure(
        prefill_chunk_size=32,
        max_tokens_per_step=1024,
        max_concurrent_requests=3,
    )

    assert updated == engine.get_config()
    assert updated.prefill_chunk_size == 32
    assert updated.max_tokens_per_step == 1024
    assert updated.max_concurrent_requests == 3
    assert engine.scheduler.max_concurrent_requests == 3
    assert engine.scheduler.free_slots == [0, 1, 2]


def test_reconfigure_prefill_chunk_size_applies_to_new_requests_only():
    torch.manual_seed(0)
    engine = _engine(prefill_chunk_size=128)

    engine.add_request([1, 2, 3, 4, 5, 6], max_new_tokens=1)
    in_flight = next(iter(engine.scheduler.waiting))

    engine.reconfigure(prefill_chunk_size=2)
    engine.add_request([10, 20, 30, 4, 5], max_new_tokens=1)
    new_request = engine.scheduler.waiting[-1]

    assert in_flight.prefill_chunk_size == 128
    assert new_request.prefill_chunk_size == 2


def test_reconfigure_max_tokens_per_step_affects_next_schedule():
    scheduler = Scheduler(max_concurrent_requests=2, max_tokens_per_step=4)
    scheduler.add_request(
        InferenceRequest(
            request_id="A",
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=2,
            prefill_chunk_size=3,
        )
    )
    scheduler.schedule()

    scheduler.reconfigure(max_tokens_per_step=2)
    result = scheduler.schedule()
    assert result.prefill_requests == []


def test_reconfigure_rejects_invalid_values():
    engine = _engine()

    for kwargs, pattern in (
        ({"prefill_chunk_size": 0}, "prefill_chunk_size"),
        ({"max_tokens_per_step": -1}, "max_tokens_per_step"),
        ({"max_concurrent_requests": 0}, "max_concurrent_requests"),
    ):
        try:
            engine.reconfigure(**kwargs)
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError as exc:
            assert pattern in str(exc)


def test_reconfigure_max_concurrent_before_cache_init():
    engine = _engine(max_concurrent_requests=2)
    engine.reconfigure(max_concurrent_requests=4)

    assert engine.max_concurrent_requests == 4
    assert engine.scheduler.free_slots == [0, 1, 2, 3]

    engine.reconfigure(max_concurrent_requests=2)
    assert engine.scheduler.free_slots == [0, 1]


def test_reconfigure_max_concurrent_decrease_after_cache_init():
    torch.manual_seed(0)
    engine = _engine(max_concurrent_requests=4)
    engine.add_request([1, 2, 3], max_new_tokens=5)
    engine.step()

    assert engine.cache is not None
    assert len(engine.scheduler.running) == 1

    engine.reconfigure(max_concurrent_requests=2)
    assert engine.max_concurrent_requests == 2
    assert engine.scheduler.free_slots == [1]


def test_reconfigure_rejects_max_concurrent_increase_after_cache_init():
    torch.manual_seed(0)
    engine = _engine(max_concurrent_requests=2)
    engine.add_request([1, 2, 3], max_new_tokens=1)
    engine.step()

    try:
        engine.reconfigure(max_concurrent_requests=4)
        raise AssertionError("expected ValueError for max_concurrent increase after cache init")
    except ValueError as exc:
        assert "cannot increase max_concurrent_requests" in str(exc)


def test_reconfigure_rejects_cache_type_toggle_after_cache_init():
    torch.manual_seed(0)
    engine = _engine(use_paged_kv_cache=True, use_prefix_cache=True)
    engine.add_request([1, 2, 3], max_new_tokens=1)
    engine.step()

    try:
        engine.reconfigure(use_paged_kv_cache=False)
        raise AssertionError("expected ValueError for use_paged_kv_cache toggle")
    except ValueError as exc:
        assert "use_paged_kv_cache" in str(exc)

    try:
        engine.reconfigure(use_prefix_cache=False)
        raise AssertionError("expected ValueError for use_prefix_cache toggle")
    except ValueError as exc:
        assert "use_prefix_cache" in str(exc)


def test_reconfigure_cache_flags_before_first_step():
    engine = _engine(use_paged_kv_cache=True, use_prefix_cache=True)
    updated = engine.reconfigure(use_paged_kv_cache=False, use_prefix_cache=False)

    assert updated.use_paged_kv_cache is False
    assert updated.use_prefix_cache is False
