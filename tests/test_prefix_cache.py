import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.gpt import GptInferenceModel
from inference.block_allocator import BlockAllocator
from inference.engine import Engine
from inference.prefix_cache import PrefixCache
from model.gpt_model import GPT_CONFIG_124M, GptModel


def test_prefix_cache_lookup_and_insert():
    allocator = BlockAllocator(8)
    cache = PrefixCache(allocator, block_size=4)

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    blocks = allocator.allocate_many(2)
    cache.insert(tokens, blocks)

    matched, shared = cache.lookup(tokens + [99])
    assert matched == 8
    assert shared == blocks


def test_prefix_cache_partial_match_stops_at_first_miss():
    allocator = BlockAllocator(4)
    cache = PrefixCache(allocator, block_size=4)

    block_a = allocator.allocate_many(1)
    cache.insert([10, 11, 12, 13], block_a)

    matched, shared = cache.lookup([10, 11, 12, 13, 20, 21, 22, 23])
    assert matched == 4
    assert shared == block_a

    matched, shared = cache.lookup([10, 11, 12, 99])
    assert matched == 0
    assert shared == []


def test_prefix_cache_ignores_partial_tail_block():
    allocator = BlockAllocator(4)
    cache = PrefixCache(allocator, block_size=4)

    block_a = allocator.allocate_many(1)
    cache.insert([1, 2, 3], block_a)

    matched, shared = cache.lookup([1, 2, 3])
    assert matched == 0
    assert shared == []


def test_prefix_cache_insert_retains_blocks():
    allocator = BlockAllocator(4)
    cache = PrefixCache(allocator, block_size=4)

    block_id = allocator.allocate()
    cache.insert([1, 2, 3, 4], [block_id])

    assert allocator.refcount(block_id) == 2


def test_prefix_cache_eviction_drops_stale_entry():
    allocator = BlockAllocator(2)
    cache = PrefixCache(allocator, block_size=4, max_entries=1)

    first_blocks = allocator.allocate_many(1)
    cache.insert([1, 2, 3, 4], first_blocks)

    second_blocks = allocator.allocate_many(1)
    cache.insert([5, 6, 7, 8], second_blocks)

    assert len(cache.entries) == 1
    matched, shared = cache.lookup([1, 2, 3, 4])
    assert matched == 0

    matched, shared = cache.lookup([5, 6, 7, 8])
    assert matched == 4
    assert shared == second_blocks


def test_prefix_cache_does_not_retain_existing_key():
    allocator = BlockAllocator(4)
    cache = PrefixCache(allocator, block_size=4)

    block_id = allocator.allocate()
    tokens = [1, 2, 3, 4]

    cache.insert(tokens, [block_id])
    assert allocator.refcount(block_id) == 2

    cache.insert(tokens, [block_id])
    assert allocator.refcount(block_id) == 2


def test_prefix_cache_init_validation():
    allocator = BlockAllocator(2)
    with pytest.raises(ValueError, match="block_size must be positive"):
        PrefixCache(allocator, block_size=0)
    with pytest.raises(ValueError, match="max_entries must be positive"):
        PrefixCache(allocator, block_size=4, max_entries=0)


def test_engine_reuses_prefix_for_shared_prompt():
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    cfg["block_size"] = 4

    device = torch.device("cpu")
    model = GptModel(cfg).to(device).eval()
    wrapped = GptInferenceModel(model)
    engine = Engine(
        wrapped,
        max_concurrent_requests=2,
        device=device,
        prefill_chunk_size=8,
        use_prefix_cache=True,
    )

    shared_prompt = list(range(8))
    engine.generate(shared_prompt + [101], max_new_tokens=1)
    first_saved = engine.cache.prefix_cache.tokens_saved

    engine.generate(shared_prompt + [202], max_new_tokens=1)
    assert engine.cache.prefix_cache.hits >= 1
    assert engine.cache.prefix_cache.tokens_saved > first_saved


def test_engine_prefix_cache_disabled():
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64

    device = torch.device("cpu")
    model = GptModel(cfg).to(device).eval()
    wrapped = GptInferenceModel(model)
    engine = Engine(
        wrapped,
        max_concurrent_requests=1,
        device=device,
        use_prefix_cache=False,
    )

    prompt = list(range(8))
    engine.generate(prompt, max_new_tokens=1)
    engine.generate(prompt, max_new_tokens=1)

    assert engine.cache.prefix_cache is None
