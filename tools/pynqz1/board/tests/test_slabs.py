from __future__ import annotations

import pytest

from board.memory.slabs import FakeSlab, fake_slabs


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
