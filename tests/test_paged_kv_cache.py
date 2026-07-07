import torch

from inference.paged_kv_cache import PagedKVCache


def test_paged_kv_cache_smoke():
    num_layers = 2
    max_seq_len = 16
    n_heads = 2
    head_dim = 4
    batch_size = 2

    cache = PagedKVCache(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
        block_size=8,
    )
    cache.init_batch(batch_size)

    key = torch.randn(3, n_heads, head_dim)
    value = torch.randn(3, n_heads, head_dim)
    cache.append(0, layer_id=0, key=key, value=value)
    assert cache.length(0) == 0, "pos advances only after the last layer"

    cache.append(0, layer_id=1, key=key, value=value)
    assert cache.length(0) == 3
    assert cache.length(1) == 0

    past_k, past_v = cache.get(0, layer_id=0)
    assert past_k.shape == (3, n_heads, head_dim)
    assert past_v.shape == (3, n_heads, head_dim)

    decode_key = torch.randn(1, n_heads, head_dim)
    decode_value = torch.randn(1, n_heads, head_dim)
    cache.append(0, layer_id=0, key=decode_key, value=decode_value)
    assert cache.length(0) == 3

    cache.append(0, layer_id=1, key=decode_key, value=decode_value)
    assert cache.length(0) == 4

    past_k, _ = cache.get(0, layer_id=0)
    assert past_k.shape == (4, n_heads, head_dim)

    key_b = torch.randn(2, n_heads, head_dim)
    value_b = torch.randn(2, n_heads, head_dim)
    cache.append(1, layer_id=0, key=key_b, value=value_b)
    cache.append(1, layer_id=1, key=key_b, value=value_b)
    assert cache.length(1) == 2
    assert torch.equal(cache.lengths(), torch.tensor([4, 2]))

    cache.free()
    assert cache.batch_size == 0
    assert cache.k is None


def test_paged_kv_cache_block_growth():
    cache = PagedKVCache(
        num_layers=2,
        max_seq_len=20,
        n_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        block_size=8,
    )
    cache.init_batch(batch_size=1)

    key = torch.randn(9, 2, 4)
    value = torch.randn(9, 2, 4)
    cache.append(0, layer_id=0, key=key, value=value)
    cache.append(0, layer_id=1, key=key, value=value)

    assert cache.length(0) == 9
    assert len(cache.block_tables[0].physical_blocks()) == 2
    assert cache.allocated_blocks == 2


def test_paged_kv_cache_reset_slot_frees_blocks():
    cache = PagedKVCache(
        num_layers=2,
        max_seq_len=16,
        n_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        block_size=8,
    )
    cache.init_batch(batch_size=1)

    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, layer_id=0, key=key, value=value)
    cache.append(0, layer_id=1, key=key, value=value)
    assert cache.allocated_blocks == 1

    freed = cache.block_tables[0].physical_blocks()[0]
    cache.reset_slot(0)
    assert cache.length(0) == 0
    assert cache.allocated_blocks == 0

    cache.append(0, layer_id=0, key=key, value=value)
    cache.append(0, layer_id=1, key=key, value=value)
    assert cache.block_tables[0].physical_blocks()[0] == freed
