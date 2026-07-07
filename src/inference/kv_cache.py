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

    def reset_slot(self, batch_idx):
        """Clear the write cursor when reusing a cache row for a new request."""
        self.pos[batch_idx] = 0

    def free(self):
        self.batch_size = 0
        self.pos = None
        self.k = None
        self.v = None
