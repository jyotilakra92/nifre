"""Physical block pool for paged KV cache."""

from __future__ import annotations


class BlockPoolExhaustedError(RuntimeError):
    """Raised when the KV block pool has no free blocks left."""


class BlockAllocator:
    """Manages a fixed pool of physical KV block IDs.

    Each ID indexes one ``(block_size, n_heads, head_dim)`` K/V slice in the
    paged cache tensor. Sequences receive IDs from this pool via ``BlockTable``.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        self._num_blocks = num_blocks
        self._free = list(range(num_blocks))
        self._in_use: set[int] = set()
        self._refcount = [0] * num_blocks

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def allocated_count(self) -> int:
        return len(self._in_use)

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def utilization(self) -> float:
        if self._num_blocks == 0:
            return 0.0
        return self.allocated_count / self._num_blocks

    def allocate(self) -> int:
        """Allocate one block. Prefer `allocate_many` for multiple blocks."""
        return self.allocate_many(1)[0]

    def allocate_many(self, count: int) -> list[int]:
        """Allocate ``count`` blocks from the free pool."""
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if count > self.free_count:
            raise BlockPoolExhaustedError(
                f"need {count} blocks, only {self.free_count} free"
            )

        allocated: list[int] = []
        for _ in range(count):
            block_id = self._free.pop()
            self._in_use.add(block_id)
            self._refcount[block_id] = 1
            allocated.append(block_id)
        return allocated

    def retain(self, block_id: int) -> None:
        """Increment a block's reference count (for shared prefix blocks)."""
        self._validate_block_id(block_id)
        if block_id not in self._in_use:
            raise ValueError(f"block {block_id} is not allocated")
        self._refcount[block_id] += 1

    def release(self, block_id: int) -> None:
        """Decrement a block's reference count; return it to the pool at zero."""
        self._validate_block_id(block_id)
        if block_id not in self._in_use:
            raise ValueError(f"block {block_id} is not allocated")
        self._refcount[block_id] -= 1
        if self._refcount[block_id] < 0:
            raise ValueError(f"block {block_id} refcount underflow")
        if self._refcount[block_id] == 0:
            self._in_use.remove(block_id)
            self._free.append(block_id)

    def refcount(self, block_id: int) -> int:
        """Return the current reference count for a block (0 if not in use)."""
        self._validate_block_id(block_id)
        return self._refcount[block_id]

    def free(self, block_id: int) -> None:
        """Release one reference to a block."""
        self.free_many([block_id])

    def free_many(self, block_ids: list[int]) -> None:
        """Release references to multiple blocks."""
        if not block_ids:
            return

        seen: set[int] = set()
        for block_id in block_ids:
            self._validate_block_id(block_id)
            if block_id in seen:
                raise ValueError(f"duplicate block_id in free_many: {block_id}")
            seen.add(block_id)

        for block_id in block_ids:
            self.release(block_id)

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self._num_blocks:
            raise ValueError(f"block_id {block_id} out of range [0, {self._num_blocks})")

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(blocks={self._num_blocks}, "
            f"allocated={self.allocated_count}, free={self.free_count})"
        )
