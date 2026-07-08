from inference.block_allocator import BlockAllocator

class BlockTable:
    """Per-sequence mapping from logical block index to physical block ID.

    Parameters
    ----------
    allocator:
        Shared block pool used to allocate and free physical blocks.
    block_size:
        Number of tokens stored in each block.
    """
    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self._allocator = allocator
        self._block_size = block_size
        self._physical_blocks: list[int] = []

    def import_blocks(self, physical_blocks: list[int]) -> None:
        """Attach shared physical blocks (e.g. from prefix cache) to this sequence."""
        if self._physical_blocks:
            raise ValueError("cannot import blocks into a non-empty block table")
        for block_id in physical_blocks:
            self._allocator.retain(block_id)
        self._physical_blocks = list(physical_blocks)

    def ensure_capacity(self, total_tokens: int) -> None:
        """Ensure blocks exist for ``total_tokens`` cached tokens (after upcoming write)."""
        if total_tokens < 0:
            raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
        blocks_needed = (total_tokens + self._block_size - 1) // self._block_size
        while len(self._physical_blocks) < blocks_needed:
            self._physical_blocks.append(self._allocator.allocate())

    def resolve(self, token_index: int) -> tuple[int, int]:
        logical_block = token_index // self._block_size
        if logical_block >= len(self._physical_blocks):
            raise IndexError(f"token_index {token_index} has no allocated block")
        return self._physical_blocks[logical_block], token_index % self._block_size

    def physical_blocks(self) -> list[int]:
        return self._physical_blocks[:]

    def clear(self) -> None:
        if self._physical_blocks:
            self._allocator.free_many(self._physical_blocks)
            self._physical_blocks.clear()