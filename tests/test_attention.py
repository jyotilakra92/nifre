import torch

from model.attention import MultiHeadAttention
from inference.kv_cache import KVCache
from inference.paged_kv_cache import PagedKVCache


def _run_attention_cache_smoke(cache):
    torch.manual_seed(0)
    d_in = d_out = 8
    num_heads = 2
    head_dim = 4
    num_layers = 2

    attn = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=16,
        dropout=0.0,
        num_heads=num_heads,
    )
    attn.eval()

    x_prefill = torch.randn(1, 3, d_in)
    out_prefill = attn(x_prefill, kv_cache=cache, layer_id=0)
    assert out_prefill.shape == (1, 3, d_out)
    assert cache.length(0) == 0, "pos advances only after the last layer"

    attn(x_prefill, kv_cache=cache, layer_id=1)
    assert cache.length(0) == 3

    past_k, past_v = cache.get(0, layer_id=0)
    assert past_k.shape == (3, num_heads, head_dim)
    assert past_v.shape == (3, num_heads, head_dim)

    x_decode = torch.randn(1, 1, d_in)
    out_decode = attn(x_decode, kv_cache=cache, layer_id=0)
    assert out_decode.shape == (1, 1, d_out)
    assert cache.length(0) == 3

    attn(x_decode, kv_cache=cache, layer_id=1)
    assert cache.length(0) == 4

    past_k, _ = cache.get(0, layer_id=0)
    assert past_k.shape == (4, num_heads, head_dim)

    cache.free()
    cache.init_batch(2)

    x_batch = torch.randn(2, 4, d_in)
    input_lens = torch.tensor([3, 4])
    attn(x_batch, kv_cache=cache, layer_id=0, input_lens=input_lens)
    attn(x_batch, kv_cache=cache, layer_id=1, input_lens=input_lens)
    assert cache.length(0) == 3
    assert cache.length(1) == 4

    x_decode_batch = torch.randn(2, 1, d_in)
    attn(x_decode_batch, kv_cache=cache, layer_id=0)
    attn(x_decode_batch, kv_cache=cache, layer_id=1)
    assert cache.length(0) == 4
    assert cache.length(1) == 5

    x_train = torch.randn(2, 4, d_in)
    out_train = attn(x_train)
    assert out_train.shape == (2, 4, d_out)


def test_attention_cache_smoke():
    cache = KVCache(
        num_layers=2,
        max_seq_len=16,
        n_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
    )
    cache.init_batch(1)
    _run_attention_cache_smoke(cache)


def test_attention_paged_cache_smoke():
    cache = PagedKVCache(
        num_layers=2,
        max_seq_len=16,
        n_heads=2,
        head_dim=4,
        device="cpu",
        dtype=torch.float32,
        block_size=8,
    )
    cache.init_batch(1)
    _run_attention_cache_smoke(cache)
