"""DDR slab implementations the allocator consumes.

A slab is a contiguous chunk of board-resident memory with three primitives:
``write``, ``read``, and ``clear``. Two concrete impls:

  * ``FakeSlab`` — a ``bytearray`` for host-side unit tests.
  * ``PynqSlab`` — a real PYNQ-allocated CMA buffer for the board daemon.

Slabs do not know about tensors. They just know about byte ranges.
"""

from __future__ import annotations

from typing import Protocol

MIB = 1024 * 1024


class Slab(Protocol):
    size: int
    physical_address: int

    def write(self, offset: int, data: bytes | memoryview) -> None: ...
    def read(self, offset: int, size: int) -> bytes: ...
    def clear(self, offset: int, size: int, value: int = 0) -> None: ...

    # PL DMA support. Default (FakeSlab) implementations are no-ops since
    # there is no cache to flush.
    def flush_range(self, offset: int, nbytes: int) -> None: ...
    def invalidate_range(self, offset: int, nbytes: int) -> None: ...

    def close(self) -> None:  # noqa: D401
        """Release any board resources held by the slab."""


class FakeSlab:
    def __init__(self, size: int, physical_address: int = 0):
        self.size = size
        self.physical_address = physical_address
        self._data = bytearray(size)

    def write(self, offset: int, data: bytes | memoryview) -> None:
        self._data[offset : offset + len(data)] = data

    def read(self, offset: int, size: int) -> bytes:
        return bytes(self._data[offset : offset + size])

    def clear(self, offset: int, size: int, value: int = 0) -> None:
        self._data[offset : offset + size] = bytes([value]) * size

    def flush_range(self, offset: int, nbytes: int) -> None:
        pass  # nothing to flush — bytearray has no cache

    def invalidate_range(self, offset: int, nbytes: int) -> None:
        pass

    def close(self) -> None:  # nothing to release
        pass


class PynqSlab:
    """Backed by a real PYNQ CMA buffer. Only imported when used."""

    def __init__(self, size: int):
        import numpy as np
        from pynq import allocate

        self.size = size
        self._np = np
        self._buf = allocate(shape=(size,), dtype=np.uint8)
        self.physical_address = int(self._buf.physical_address)

    @property
    def pynq_buffer(self):
        """Underlying PYNQ buffer for direct DMA. Slice it for partial transfers."""
        return self._buf

    def write(self, offset: int, data: bytes | memoryview) -> None:
        arr = self._np.frombuffer(data, dtype=self._np.uint8)
        self._buf[offset : offset + len(data)] = arr
        self._buf.flush()

    def read(self, offset: int, size: int) -> bytes:
        self._buf.invalidate()
        return bytes(self._buf[offset : offset + size])

    def clear(self, offset: int, size: int, value: int = 0) -> None:
        self._buf[offset : offset + size] = value
        self._buf.flush()

    def flush_range(self, offset: int, nbytes: int) -> None:
        # PYNQ's per-slice flush is not always reliable across drivers;
        # whole-buffer flush is cheap at our slab sizes (≤32 MiB).
        del offset, nbytes
        self._buf.flush()

    def invalidate_range(self, offset: int, nbytes: int) -> None:
        del offset, nbytes
        self._buf.invalidate()

    def close(self) -> None:
        self._buf.freebuffer()


def fake_slabs(total_bytes: int, slab_bytes: int) -> list[FakeSlab]:
    if total_bytes <= 0 or slab_bytes <= 0:
        raise ValueError("total_bytes and slab_bytes must be positive")
    slabs: list[FakeSlab] = []
    base_address = 0x1000_0000
    remaining = total_bytes
    while remaining:
        size = min(slab_bytes, remaining)
        slabs.append(FakeSlab(size=size, physical_address=base_address))
        base_address += size + 0x1000
        remaining -= size
    return slabs


def pynq_slabs(total_bytes: int, slab_bytes: int) -> list[PynqSlab]:
    if total_bytes <= 0 or slab_bytes <= 0:
        raise ValueError("total_bytes and slab_bytes must be positive")
    slabs: list[PynqSlab] = []
    remaining = total_bytes
    try:
        while remaining:
            size = min(slab_bytes, remaining)
            slabs.append(PynqSlab(size=size))
            remaining -= size
        return slabs
    except Exception:
        for slab in slabs:
            slab.close()
        raise
