import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.backends.registry import load_backend
from inference.engine import Engine


def _load_gpt2(device, context_length=128):
    return load_backend(
        "hf",
        checkpoint=None,
        device=device,
        hf_model="gpt2",
        context_length=context_length,
    )


def test_hf_auto_engine_generate():
    device = torch.device("cpu")
    model, tokenizer = _load_gpt2(device, context_length=64)
    assert getattr(model, "supports_paged_kv_cache", True) is False
    assert getattr(model, "supports_prefix_cache", False) is True
    assert model.config.num_layers == 12
    engine = Engine(model, max_concurrent_requests=2, device=device)
    prompt = tokenizer.encode("Hello")
    result = engine.generate(prompt, max_new_tokens=3)
    assert result.state.value == "finished"
    assert len(result.output_token_ids) == 3


def test_hf_concurrent_prefills_of_different_lengths():
    """Batched prefill rows of differing lengths must concatenate (regression)."""
    device = torch.device("cpu")
    model, tokenizer = _load_gpt2(device)
    engine = Engine(model, max_concurrent_requests=4, device=device)

    ids_a = engine.add_request(tokenizer.encode("A short prompt"), 3)
    ids_b = engine.add_request(
        tokenizer.encode("A noticeably longer prompt with several more tokens here"),
        3,
    )
    # Drive both concurrently through the same batched prefill step.
    while (
        ids_a not in engine.scheduler.completed
        or ids_b not in engine.scheduler.completed
    ):
        engine.step()

    assert len(engine.scheduler.completed[ids_a].output_token_ids) == 3
    assert len(engine.scheduler.completed[ids_b].output_token_ids) == 3


def test_hf_prefix_cache_reuses_shared_prefix_without_changing_output():
    device = torch.device("cpu")
    model, tokenizer = _load_gpt2(device)

    shared = (
        "The quick brown fox jumps over the lazy dog while the sun sets "
        "slowly behind the distant mountains and the river keeps flowing"
    )
    prompt_a = tokenizer.encode(shared)
    prompt_b = tokenizer.encode(shared + " toward the calm and quiet sea.")
    assert len(prompt_a) > 16  # long enough to exceed min_prefix_tokens

    # Reference output with prefix caching disabled.
    engine_off = Engine(
        model, max_concurrent_requests=2, device=device, use_prefix_cache=False
    )
    ref = engine_off.generate(prompt_b, max_new_tokens=5)
    assert ref.prefix_cache_hit_tokens == 0

    # Same model, prefix caching enabled: warm the cache with A, then run B.
    engine_on = Engine(
        model, max_concurrent_requests=2, device=device, use_prefix_cache=True
    )
    engine_on.generate(prompt_a, max_new_tokens=1)
    hit = engine_on.generate(prompt_b, max_new_tokens=5)

    assert hit.prefix_cache_hit_tokens > 0
    assert hit.prefix_cache_hit_tokens < len(prompt_b)
    # Block-aligned reuse: a whole number of blocks.
    block_size = model.config.block_size
    assert hit.prefix_cache_hit_tokens % block_size == 0
    # Reusing cached K/V must not change the generated tokens.
    assert hit.output_token_ids == ref.output_token_ids


def test_hf_prefix_cache_metrics_and_block_dedup():
    device = torch.device("cpu")
    model, tokenizer = _load_gpt2(device)
    engine = Engine(
        model, max_concurrent_requests=2, device=device, use_prefix_cache=True
    )

    shared = (
        "System: you are a concise assistant that answers questions about "
        "distributed systems, caching, and inference engines in one sentence. "
    )
    engine.generate(tokenizer.encode(shared + "Question one?"), max_new_tokens=1)
    stats_after_first = engine.cache.prefix_stats()
    entries_after_first = stats_after_first["entries"]
    mem_after_first = stats_after_first["memory_mb"]
    assert entries_after_first > 0
    assert mem_after_first > 0

    engine.generate(tokenizer.encode(shared + "A different question two?"), max_new_tokens=1)
    stats = engine.cache.prefix_stats()

    assert stats["hits"] >= 1
    assert stats["tokens_reused"] > 0
    assert stats["avg_lookup_ms"] >= 0.0
    # Shared blocks are stored once: the second prompt should add few (if any)
    # new blocks and never double-count the shared prefix's memory.
    assert stats["entries"] < entries_after_first * 2
