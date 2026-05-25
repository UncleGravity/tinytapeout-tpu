"""TensorAllocator.slab_view — single-extent DMA resolution."""

from __future__ import annotations

import pytest

from board.memory.allocator import AllocatorError, TensorAllocator
from board.memory.slabs import fake_slabs


def make_allocator(total: int = 8 * 1024, slab: int = 4 * 1024) -> TensorAllocator:
    return TensorAllocator(fake_slabs(total, slab))


def test_slab_view_single_extent_resolves_to_offset_in_slab():
    alloc = make_allocator()
    record = alloc.allocate(256)
    slab, abs_off, n = alloc.slab_view(record.handle, 16, 64)
    assert slab is alloc._slabs[record.extents[0].slab_index]
    assert abs_off == record.extents[0].offset + 16
    assert n == 64


def test_slab_view_raises_on_cross_slab_range():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    record = alloc.allocate(6 * 1024)  # forced to span two slabs
    assert len(record.extents) >= 2
    with pytest.raises(AllocatorError) as exc:
        alloc.slab_view(record.handle, 0, 6 * 1024)
    assert exc.value.code == "multi_extent"


def test_slab_view_within_first_extent_of_multi_extent_is_ok():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    record = alloc.allocate(6 * 1024)
    # First extent is at least 4 KiB worth of bytes; a short range inside it
    # must succeed even though the tensor as a whole spans slabs.
    first_extent_size = record.extents[0].nbytes
    slab, abs_off, n = alloc.slab_view(record.handle, 0, min(64, first_extent_size))
    assert n == min(64, first_extent_size)
    assert slab is alloc._slabs[record.extents[0].slab_index]


def test_slab_view_validates_bounds():
    alloc = make_allocator()
    record = alloc.allocate(64)
    with pytest.raises(AllocatorError) as exc:
        alloc.slab_view(record.handle, 0, 128)
    assert exc.value.code == "out_of_bounds"
