import torch


class KVCache:
    """Batched key/value cache for autoregressive transformer inference.

    One pre-allocated K/V tensor per layer, shape
    ``(batch_size, max_seq_len, n_heads, head_dim)``. Each batch row has its
    own write cursor in ``pos`` (shape ``(batch_size,)``).
    """

    def __init__(self, num_layers, max_seq_len, n_heads, head_dim, device="cpu", dtype=torch.float16):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.batch_size = 0
        self.pos = None
        self.k = None
        self.v = None

    def init_batch(self, batch_size):
        self.batch_size = batch_size
        self.pos = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.k = [
            torch.empty(
                batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]
        self.v = [
            torch.empty(
                batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.num_layers)
        ]

    def append(self, batch_idx, layer_id, key, value):
        pos = self.pos[batch_idx].item()
        n = key.shape[0]
        assert pos + n <= self.max_seq_len

        self.k[layer_id][batch_idx, pos : pos + n] = key
        self.v[layer_id][batch_idx, pos : pos + n] = value

        if layer_id == self.num_layers - 1:
            self.pos[batch_idx] += n

    def get(self, batch_idx, layer_id):
        pos = self.pos[batch_idx].item()
        return self.k[layer_id][batch_idx, :pos], self.v[layer_id][batch_idx, :pos]

    def length(self, batch_idx):
        return self.pos[batch_idx].item()

    def lengths(self):
        return self.pos.clone()

    def free(self):
        self.batch_size = 0
        self.pos = None
        self.k = None
        self.v = None


def _smoke_test():
    num_layers = 2
    max_seq_len = 16
    n_heads = 2
    head_dim = 4
    batch_size = 2

    cache = KVCache(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
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

    print("kv_cache smoke test passed")


if __name__ == "__main__":
    _smoke_test()
