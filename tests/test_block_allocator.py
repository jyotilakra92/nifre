import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.block_allocator import BlockAllocator, BlockPoolExhaustedError


def test_init_rejects_invalid_size():
    with pytest.raises(ValueError, match="num_blocks must be positive"):
        BlockAllocator(0)


def test_allocate_and_free():
    allocator = BlockAllocator(4)

    blocks = allocator.allocate_many(2)
    assert len(blocks) == 2
    assert allocator.allocated_count == 2
    assert allocator.free_count == 2

    allocator.free_many(blocks)
    assert allocator.allocated_count == 0
    assert allocator.free_count == 4


def test_single_allocate_wrapper():
    allocator = BlockAllocator(2)
    block_id = allocator.allocate()
    assert block_id in range(2)
    allocator.free(block_id)


def test_block_reuse_after_free():
    allocator = BlockAllocator(2)

    first = allocator.allocate()
    allocator.free(first)

    second = allocator.allocate()
    assert second == first


def test_pool_exhausted():
    allocator = BlockAllocator(1)
    allocator.allocate()

    with pytest.raises(BlockPoolExhaustedError, match="only 0 free"):
        allocator.allocate_many(1)


def test_free_out_of_range_rejected():
    allocator = BlockAllocator(2)
    with pytest.raises(ValueError, match="out of range"):
        allocator.free(99)


def test_double_free_rejected():
    allocator = BlockAllocator(2)
    block_id = allocator.allocate()
    allocator.free(block_id)

    with pytest.raises(ValueError, match="not allocated"):
        allocator.free(block_id)


def test_free_many_duplicate_rejected():
    allocator = BlockAllocator(2)
    a, b = allocator.allocate(), allocator.allocate()

    with pytest.raises(ValueError, match="duplicate"):
        allocator.free_many([a, a])


def test_utilization():
    allocator = BlockAllocator(4)
    assert allocator.utilization == 0.0

    allocator.allocate_many(1)
    assert allocator.utilization == 0.25


def test_free_many_empty_list():
    allocator = BlockAllocator(2)
    allocator.free_many([])


def test_allocate_many_rejects_invalid_count():
    allocator = BlockAllocator(2)
    with pytest.raises(ValueError, match="count must be positive"):
        allocator.allocate_many(0)
