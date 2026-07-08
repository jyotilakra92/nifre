import torch

from inference.block_allocator import BlockAllocator
from inference.block_table import BlockTable
from inference.prefix_cache import PrefixCache


class PagedKVCache:
    """Paged key/value cache using a shared block pool and per-slot block tables.

    Physical storage per layer:
    ``(num_blocks, block_size, n_heads, head_dim)``

    The public API matches :class:`inference.kv_cache.KVCache`.
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
        block_size: int = 16,
        enable_prefix_cache: bool = True,
        prefix_cache_max_entries: int = 1024,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache_max_entries = prefix_cache_max_entries

        self.batch_size = 0
        self.num_blocks = 0
        self.pos = None
        self.k = None
        self.v = None
        self.block_allocator: BlockAllocator | None = None
        self.block_tables: list[BlockTable] = []
        self.prefix_cache: PrefixCache | None = None

    def init_batch(self, batch_size: int) -> None:
        """Initialize the cache for a new batch of sequences."""
        self.batch_size = batch_size
        self.pos = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        blocks_per_slot = (self.max_seq_len + self.block_size - 1) // self.block_size
        self.num_blocks = blocks_per_slot * batch_size
        self.block_allocator = BlockAllocator(self.num_blocks)
        self.block_tables = [
            BlockTable(self.block_allocator, self.block_size) for _ in range(batch_size)
        ]
        self.prefix_cache = (
            PrefixCache(
                self.block_allocator,
                self.block_size,
                max_entries=self.prefix_cache_max_entries,
            )
            if self.enable_prefix_cache
            else None
        )

        block_shape = (
            self.num_blocks,
            self.block_size,
            self.n_heads,
            self.head_dim,
        )
        self.k = [
            torch.empty(block_shape, device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self.v = [
            torch.empty(block_shape, device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]

    def try_load_prefix(self, batch_idx: int, token_ids: list[int]) -> int:
        """Load a cached prefix into a slot. Returns the number of tokens restored."""
        if self.prefix_cache is None:
            return 0

        num_tokens, physical_blocks = self.prefix_cache.lookup(token_ids)
        if num_tokens == 0:
            return 0

        self.block_tables[batch_idx].import_blocks(physical_blocks)
        self.pos[batch_idx] = num_tokens
        return num_tokens

    def register_prefix(self, batch_idx: int, token_ids: list[int]) -> None:
        """Store a completed prompt prefix in the shared cache."""
        if self.prefix_cache is None:
            return
        physical_blocks = self.block_tables[batch_idx].physical_blocks()
        self.prefix_cache.insert(token_ids, physical_blocks)

    def append(self, batch_idx: int, layer_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        pos = self.pos[batch_idx].item()
        n = key.shape[0]
        assert pos + n <= self.max_seq_len

        table = self.block_tables[batch_idx]
        table.ensure_capacity(pos + n)

        for token_offset in range(n):
            physical_block_id, offset_in_block = table.resolve(pos + token_offset)
            self.k[layer_id][physical_block_id, offset_in_block] = key[token_offset]
            self.v[layer_id][physical_block_id, offset_in_block] = value[token_offset]

        if layer_id == self.num_layers - 1:
            self.pos[batch_idx] += n

    def get(self, batch_idx: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.pos[batch_idx].item()
        if pos == 0:
            empty = torch.empty(
                0, self.n_heads, self.head_dim, device=self.device, dtype=self.dtype
            )
            return empty, empty.clone()

        keys = []
        values = []
        remaining = pos
        for physical_block_id in self.block_tables[batch_idx].physical_blocks():
            if remaining <= 0:
                break
            take = min(remaining, self.block_size)
            keys.append(self.k[layer_id][physical_block_id, :take])
            values.append(self.v[layer_id][physical_block_id, :take])
            remaining -= take

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def length(self, batch_idx: int) -> int:
        return self.pos[batch_idx].item()

    def lengths(self) -> torch.Tensor:
        return self.pos.clone()

    def reset_slot(self, batch_idx: int) -> None:
        """Free a slot's blocks and reset its write cursor."""
        self.block_tables[batch_idx].clear()
        self.pos[batch_idx] = 0

    @property
    def allocated_blocks(self) -> int:
        if self.block_allocator is None:
            return 0
        return self.block_allocator.allocated_count

    def free(self) -> None:
        self.batch_size = 0
        self.num_blocks = 0
        self.pos = None
        self.k = None
        self.v = None
        self.block_allocator = None
        self.block_tables = []
        self.prefix_cache = None
