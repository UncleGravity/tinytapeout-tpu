from __future__ import annotations

import pytest

from board.memory.slabs import CACHELINE, FakeSlab, PynqSlab, fake_slabs


def test_fake_slab_round_trip():
    slab = FakeSlab(size=128)
    slab.write(16, b"hello world")
    assert slab.read(16, 11) == b"hello world"


def test_fake_slab_clear():
    slab = FakeSlab(size=8)
    slab.write(0, b"\xff" * 8)
    slab.clear(2, 4, value=0xAA)
    assert slab.read(0, 8) == b"\xff\xff" + b"\xaa" * 4 + b"\xff\xff"


def test_fake_slabs_splits_total_into_chunks():
    slabs = fake_slabs(total_bytes=300, slab_bytes=128)
    assert [s.size for s in slabs] == [128, 128, 44]
    # Physical addresses are monotonically increasing with a small gap.
    addrs = [s.physical_address for s in slabs]
    assert addrs == sorted(addrs)
    assert addrs[1] > addrs[0] + slabs[0].size


def test_fake_slabs_rejects_zero():
    with pytest.raises(ValueError):
        fake_slabs(total_bytes=0, slab_bytes=128)


# -- PynqSlab range-flush bounds (the cache-coherence-critical math) --------
# PynqSlab itself needs a board, but _cacheline_bounds is pure arithmetic.


def test_cacheline_bounds_empty_range_is_none():
    assert PynqSlab._cacheline_bounds(0, 0, 1024) is None
    assert PynqSlab._cacheline_bounds(64, -1, 1024) is None


def test_cacheline_bounds_rounds_out_to_whole_lines():
    # offset 40, len 10 → [40,50) must round to [32, 64): covers both partial
    # lines so no dirty/stale byte is left behind.
    assert PynqSlab._cacheline_bounds(40, 10, 1024) == (32, 64)


def test_cacheline_bounds_aligned_range_unchanged():
    assert PynqSlab._cacheline_bounds(CACHELINE, CACHELINE, 1024) == (
        CACHELINE, 2 * CACHELINE)


def test_cacheline_bounds_clamps_end_to_size():
    size = 100
    start, end = PynqSlab._cacheline_bounds(96, 8, size)
    assert start == 96  # already line-aligned
    assert end == size  # would be 128, clamped to slab size


def test_cacheline_bounds_always_covers_requested_range():
    # Property: the returned window always fully contains [offset, offset+n).
    for offset in range(0, 200):
        for nbytes in range(1, 80):
            bounds = PynqSlab._cacheline_bounds(offset, nbytes, 4096)
            assert bounds is not None
            start, end = bounds
            assert start <= offset
            assert end >= offset + nbytes
            assert start % CACHELINE == 0
