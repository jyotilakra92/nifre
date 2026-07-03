import torch

class KVCache:
    """Per-request key/value cache for autoregressive transformer inference.

    Stores precomputed attention keys and values so each new token only
    attends over cached past tokens instead of recomputing the full sequence.
    """

    def __init__(self, num_layers, max_seq_len, n_heads, head_dim, device="cpu", dtype=torch.float16):
        """Configure shared cache dimensions and storage settings.

        Args:
            num_layers: Number of transformer layers; one K/V tensor pair per layer.
            max_seq_len: Maximum sequence length (tokens) each request may cache.
            n_heads: Number of attention heads per layer.
            head_dim: Dimension of each attention head.
            device: Torch device for allocated cache tensors (e.g. "cpu", "cuda:0").
            dtype: Floating-point dtype for cache tensors (default: float16).
        """
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        # request_id -> {"pos", "k", "v"} entry (see init_request)
        self.cache = {}

    def init_request(self, request_id):
        """Allocate empty K/V buffers for a new inference request.

        Args:
            request_id: Unique identifier for the request (e.g. session or batch slot).
        """
        self.cache[request_id] = {
            "pos": 0,
            "k": [
                torch.empty(
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ],
            "v": [
                torch.empty(
                    self.max_seq_len,
                    self.n_heads,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ],
        }

    def append(self, request_id, layer_id, key, value):
        """Write new key/value vectors into the cache for one layer.

        Tokens are appended at the current write position. The position counter
        advances only after the last layer is updated, so all layers stay aligned
        for the same token span.

        Args:
            request_id: Request whose cache to update.
            layer_id: Index of the transformer layer (0 .. num_layers - 1).
            key: Key tensor of shape (n_tokens, n_heads, head_dim).
            value: Value tensor of shape (n_tokens, n_heads, head_dim).
        """
        entry = self.cache[request_id]
        pos = entry["pos"]

        n = key.shape[0]
        assert pos + n <= self.max_seq_len

        entry["k"][layer_id][pos : pos + n] = key
        entry["v"][layer_id][pos : pos + n] = value

        if layer_id == self.num_layers - 1:
            entry["pos"] += n

    def get(self, request_id, layer_id):
        """Return cached keys and values for a layer up to the current length.

        Args:
            request_id: Request whose cache to read.
            layer_id: Index of the transformer layer.

        Returns:
            Tuple (keys, values), each of shape (cached_len, n_heads, head_dim).
        """
        entry = self.cache[request_id]
        pos = entry["pos"]
        return entry["k"][layer_id][:pos], entry["v"][layer_id][:pos]

    def length(self, request_id):
        """Return the number of tokens currently stored for a request.

        Args:
            request_id: Request whose cached sequence length to query.

        Returns:
            Number of cached token positions (0 if the request was just initialized).
        """
        return self.cache[request_id]["pos"]

    def free(self, request_id):
        """Release cache memory for a finished or cancelled request.

        Args:
            request_id: Request to remove from the cache.
        """
        del self.cache[request_id]
