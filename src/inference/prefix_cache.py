"""Block-chained prefix cache for reusing KV blocks across requests."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference.block_allocator import BlockAllocator

CacheKey = tuple[int, tuple[int, ...]]


class PrefixCache:
    """Maps chained token-block hashes to physical KV block IDs."""

    ROOT_HASH = 0

    def __init__(self, block_allocator: BlockAllocator, block_size: int, max_entries: int = 1024) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")

        self.block_allocator = block_allocator
        self.block_size = block_size
        self.max_entries = max_entries
        self.entries: OrderedDict[CacheKey, int] = OrderedDict()

    def lookup(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Return ``(matched_tokens, physical_blocks)`` for the longest cached prefix."""
        parent_hash = self.ROOT_HASH
        matched_tokens = 0
        matched_blocks: list[int] = []
        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            key = (parent_hash, block_tokens)
            block_id = self.entries.get(key)
            if block_id is None:
                break

            matched_blocks.append(block_id)
            matched_tokens += self.block_size
            self.entries.move_to_end(key)
            parent_hash = self._chain(parent_hash, block_tokens)

        return matched_tokens, matched_blocks

    def insert(self, token_ids: list[int], physical_blocks: list[int]) -> None:
        """Register full prompt blocks in the cache, retaining each new entry's block."""
        parent_hash = self.ROOT_HASH
        num_full_blocks = len(token_ids) // self.block_size

        for block_idx in range(num_full_blocks):
            if block_idx >= len(physical_blocks):
                break

            start = block_idx * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            key = (parent_hash, block_tokens)
            block_id = physical_blocks[block_idx]

            if key not in self.entries:
                while len(self.entries) >= self.max_entries:
                    self._evict_oldest()
                self.entries[key] = block_id
                self.block_allocator.retain(block_id)

            self.entries.move_to_end(key)
            parent_hash = self._chain(parent_hash, block_tokens)

    def _evict_oldest(self) -> None:
        _key, block_id = self.entries.popitem(last=False)
        self.block_allocator.release(block_id)

    @staticmethod
    def _chain(parent_hash: int, block_tokens: tuple[int, ...]) -> int:
        return hash((parent_hash, block_tokens))
