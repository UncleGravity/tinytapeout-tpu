from __future__ import annotations

import pytest

from board.memory.allocator import AllocatorError, TensorAllocator
from board.memory.slabs import fake_slabs


def make_allocator(total: int = 8 * 1024, slab: int = 4 * 1024) -> TensorAllocator:
    return TensorAllocator(fake_slabs(total, slab))


def test_allocate_returns_handle_and_record():
    alloc = make_allocator()
    record = alloc.allocate(1024, shape=[16, 16], dtype="f32", usage="weight")
    assert record.handle == 1
    assert record.nbytes == 1024
    assert record.shape == (16, 16)
    assert record.dtype == "f32"
    assert alloc.used_bytes == 1024


def test_allocate_alignment_is_respected():
    alloc = make_allocator()
    alloc.allocate(7)  # consume a bit
    record = alloc.allocate(64, alignment=64)
    assert all(e.offset % 64 == 0 for e in record.extents)


def test_allocate_spans_slabs_when_no_single_slab_fits():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    record = alloc.allocate(5 * 1024)
    assert record.nbytes == 5 * 1024
    assert len(record.extents) >= 2  # forced to span slabs
    assert sum(e.nbytes for e in record.extents) == 5 * 1024


def test_allocate_fails_if_larger_than_total_free():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    with pytest.raises(AllocatorError) as exc:
        alloc.allocate(9 * 1024)
    assert exc.value.code == "out_of_memory"
    # Free pool still intact after a failed allocation.
    assert alloc.used_bytes == 0


def test_cross_slab_round_trip():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    record = alloc.allocate(6 * 1024)
    payload = bytes((i * 31 + 7) % 251 for i in range(6 * 1024))
    alloc.write(record.handle, 0, payload)
    assert alloc.read(record.handle, 0, 6 * 1024) == payload
    # Read a slice that straddles the extent boundary.
    assert alloc.read(record.handle, 3 * 1024, 2 * 1024) == payload[3 * 1024 : 5 * 1024]


def test_extents_exposed_for_pl_dma():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    record = alloc.allocate(6 * 1024)
    extents = alloc.extents(record.handle)
    assert sum(e.nbytes for e in extents) == 6 * 1024
    # Each extent has a real physical address.
    addrs = [
        alloc.physical(record.handle, sum(e.nbytes for e in extents[:i]))
        for i in range(len(extents))
    ]
    assert len(set(addrs)) == len(extents)


def test_physical_rejects_offset_past_end():
    alloc = make_allocator()
    record = alloc.allocate(64)
    with pytest.raises(AllocatorError):
        alloc.physical(record.handle, 65)


def test_free_returns_bytes_to_pool():
    alloc = make_allocator()
    record = alloc.allocate(1024)
    alloc.free(record.handle)
    assert alloc.used_bytes == 0
    # Subsequent large allocation succeeds again.
    alloc.allocate(4 * 1024)


def test_write_read_round_trip():
    alloc = make_allocator()
    record = alloc.allocate(256)
    alloc.write(record.handle, 0, b"\xde\xad\xbe\xef" * 64)
    assert alloc.read(record.handle, 0, 8) == b"\xde\xad\xbe\xef\xde\xad\xbe\xef"


def test_write_outside_tensor_raises():
    alloc = make_allocator()
    record = alloc.allocate(64)
    with pytest.raises(AllocatorError) as exc:
        alloc.write(record.handle, 0, b"\x00" * 128)
    assert exc.value.code == "out_of_bounds"


def test_unknown_handle_raises():
    alloc = make_allocator()
    with pytest.raises(AllocatorError) as exc:
        alloc.read(999, 0, 1)
    assert exc.value.code == "unknown_tensor"


def test_physical_address_includes_slab_offset():
    alloc = make_allocator(total=8 * 1024, slab=4 * 1024)
    first = alloc.allocate(2048)
    second = alloc.allocate(2048)
    assert alloc.physical(first.handle) < alloc.physical(second.handle)


def test_zero_byte_allocation_returns_handle():
    alloc = make_allocator()
    record = alloc.allocate(0)
    assert record.nbytes == 0
    alloc.write(record.handle, 0, b"")
    assert alloc.read(record.handle, 0, 0) == b""
