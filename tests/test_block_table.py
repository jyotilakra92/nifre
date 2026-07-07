import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.block_allocator import BlockAllocator, BlockPoolExhaustedError
from inference.block_table import BlockTable


def _table(block_size: int = 16, num_blocks: int = 8) -> BlockTable:
    return BlockTable(BlockAllocator(num_blocks), block_size)


def test_init_rejects_invalid_block_size():
    allocator = BlockAllocator(4)
    with pytest.raises(ValueError, match="block_size must be positive"):
        BlockTable(allocator, 0)


def test_ensure_capacity_zero_tokens():
    table = _table()
    table.ensure_capacity(0)
    assert table.physical_blocks() == []


def test_ensure_capacity_single_block():
    table = _table(block_size=16)
    table.ensure_capacity(5)
    assert len(table.physical_blocks()) == 1


def test_ensure_capacity_idempotent():
    table = _table(block_size=16)
    table.ensure_capacity(5)
    blocks = table.physical_blocks()

    table.ensure_capacity(5)
    assert table.physical_blocks() == blocks


def test_ensure_capacity_grows_with_sequence():
    table = _table(block_size=16)
    table.ensure_capacity(5)
    assert len(table.physical_blocks()) == 1

    table.ensure_capacity(17)
    assert len(table.physical_blocks()) == 2


def test_ensure_capacity_rejects_negative_total():
    table = _table()
    with pytest.raises(ValueError, match="total_tokens must be non-negative"):
        table.ensure_capacity(-1)


def test_resolve_first_and_second_block():
    table = _table(block_size=16)
    table.ensure_capacity(17)

    physical_0, offset_0 = table.resolve(0)
    physical_16, offset_16 = table.resolve(16)

    assert offset_0 == 0
    assert offset_16 == 0
    assert physical_0 == table.physical_blocks()[0]
    assert physical_16 == table.physical_blocks()[1]


def test_resolve_offset_within_block():
    table = _table(block_size=16)
    table.ensure_capacity(10)

    physical, offset = table.resolve(9)
    assert physical == table.physical_blocks()[0]
    assert offset == 9


def test_resolve_unallocated_token_raises():
    table = _table(block_size=16)
    table.ensure_capacity(5)

    with pytest.raises(IndexError, match="no allocated block"):
        table.resolve(16)


def test_clear_frees_blocks_to_allocator():
    allocator = BlockAllocator(4)
    table = BlockTable(allocator, block_size=16)

    table.ensure_capacity(5)
    blocks = table.physical_blocks()
    assert allocator.allocated_count == 1

    table.clear()
    assert table.physical_blocks() == []
    assert allocator.allocated_count == 0
    assert allocator.free_count == 4
    for block_id in blocks:
        assert block_id in range(4)


def test_clear_allows_block_reuse():
    allocator = BlockAllocator(2)
    table = BlockTable(allocator, block_size=16)

    table.ensure_capacity(5)
    freed = table.physical_blocks()[0]
    table.clear()

    table.ensure_capacity(5)
    assert table.physical_blocks()[0] == freed


def test_pool_exhausted_propagates():
    table = _table(block_size=1, num_blocks=1)
    table.ensure_capacity(1)

    with pytest.raises(BlockPoolExhaustedError):
        table.ensure_capacity(2)
