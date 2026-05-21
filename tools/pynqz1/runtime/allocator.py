from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MIB = 1024 * 1024


class AllocatorError(RuntimeError):
    """Allocator failure that can be returned over RPC as a stable error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Extent:
    slab_index: int
    offset: int
    nbytes: int


@dataclass(frozen=True)
class TensorRecord:
    handle: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    usage: str
    layout: str
    extents: tuple[Extent, ...]


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


class FakeSlab:
    def __init__(self, size: int, physical_address: int):
        self.size = size
        self.physical_address = physical_address
        self._data = bytearray(size)

    def write(self, offset: int, data: bytes | memoryview) -> None:
        end = offset + len(data)
        self._data[offset:end] = data

    def read(self, offset: int, size: int) -> bytes:
        return bytes(self._data[offset : offset + size])

    def clear(self, offset: int, size: int, value: int = 0) -> None:
        self._data[offset : offset + size] = bytes([value]) * size


class PynqSlab:
    def __init__(self, size: int):
        import numpy as np
        from pynq import allocate

        self.size = size
        self._np = np
        self._buf = allocate(shape=(size,), dtype=np.uint8)
        self.physical_address = int(self._buf.physical_address)

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

    def close(self) -> None:
        self._buf.freebuffer()


class TensorAllocator:
    def __init__(self, slabs: Iterable[FakeSlab | PynqSlab], default_alignment: int = 64):
        self._slabs = list(slabs)
        if not self._slabs:
            raise ValueError("at least one slab is required")

        self._free: list[list[tuple[int, int]]] = [
            [(0, slab.size)] for slab in self._slabs
        ]
        self._records: dict[int, TensorRecord] = {}
        self._next_handle = 1
        self.default_alignment = default_alignment

    @property
    def total_bytes(self) -> int:
        return sum(slab.size for slab in self._slabs)

    @property
    def free_bytes(self) -> int:
        return sum(size for slab_free in self._free for _, size in slab_free)

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.free_bytes

    def memory_info(self) -> dict[str, int]:
        return {
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "used_bytes": self.used_bytes,
            "slab_count": len(self._slabs),
            "tensor_count": len(self._records),
        }

    def allocate(
        self,
        nbytes: int,
        *,
        shape: Iterable[int] = (),
        dtype: str = "u8",
        usage: str = "tensor",
        layout: str = "raw",
        alignment: int | None = None,
    ) -> TensorRecord:
        if nbytes < 0:
            raise AllocatorError("invalid_request", "nbytes must be non-negative")

        align = alignment or self.default_alignment
        remaining = nbytes
        extents: list[Extent] = []

        for slab_index in range(len(self._slabs)):
            while remaining > 0:
                reserved = self._reserve_from_slab(slab_index, remaining, align)
                if reserved is None:
                    break

                offset, extent_size = reserved
                extents.append(Extent(slab_index, offset, extent_size))
                remaining -= extent_size

        if remaining != 0:
            for extent in extents:
                self._release_extent(extent)
            raise AllocatorError(
                "out_of_memory",
                f"cannot allocate {nbytes} bytes with {self.free_bytes} bytes free",
            )

        handle = self._next_handle
        self._next_handle += 1
        record = TensorRecord(
            handle=handle,
            nbytes=nbytes,
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
            usage=usage,
            layout=layout,
            extents=tuple(extents),
        )
        self._records[handle] = record
        return record

    def free(self, handle: int) -> TensorRecord:
        record = self._get_record(handle)
        del self._records[handle]
        for extent in record.extents:
            self._release_extent(extent)
        return record

    def write(self, handle: int, offset: int, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data)
        self._validate_range(handle, offset, len(view))
        src_offset = 0
        for extent, local_offset, count in self._walk(handle, offset, len(view)):
            slab = self._slabs[extent.slab_index]
            slab.write(extent.offset + local_offset, view[src_offset : src_offset + count])
            src_offset += count

    def read(self, handle: int, offset: int, size: int) -> bytes:
        self._validate_range(handle, offset, size)
        parts = []
        for extent, local_offset, count in self._walk(handle, offset, size):
            slab = self._slabs[extent.slab_index]
            parts.append(slab.read(extent.offset + local_offset, count))
        return b"".join(parts)

    def describe(self, handle: int) -> dict[str, object]:
        record = self._get_record(handle)
        return describe_record(record)

    def close(self) -> None:
        for slab in self._slabs:
            close = getattr(slab, "close", None)
            if close is not None:
                close()

    def _reserve_from_slab(
        self, slab_index: int, max_size: int, alignment: int
    ) -> tuple[int, int] | None:
        slab_free = self._free[slab_index]
        for block_index, (offset, size) in enumerate(slab_free):
            aligned = align_up(offset, alignment)
            padding = aligned - offset
            available = size - padding
            if available <= 0:
                continue

            take = min(max_size, available)
            replacement = []
            if padding:
                replacement.append((offset, padding))
            tail_offset = aligned + take
            tail_size = size - padding - take
            if tail_size:
                replacement.append((tail_offset, tail_size))

            slab_free[block_index : block_index + 1] = replacement
            return aligned, take

        return None

    def _release_extent(self, extent: Extent) -> None:
        slab_free = self._free[extent.slab_index]
        slab_free.append((extent.offset, extent.nbytes))
        slab_free.sort()

        merged: list[tuple[int, int]] = []
        for offset, size in slab_free:
            if not merged:
                merged.append((offset, size))
                continue

            prev_offset, prev_size = merged[-1]
            prev_end = prev_offset + prev_size
            if offset <= prev_end:
                merged[-1] = (prev_offset, max(prev_end, offset + size) - prev_offset)
            else:
                merged.append((offset, size))

        self._free[extent.slab_index] = merged

    def _get_record(self, handle: int) -> TensorRecord:
        try:
            return self._records[int(handle)]
        except (KeyError, ValueError):
            raise AllocatorError("unknown_tensor", f"unknown tensor handle {handle}") from None

    def _validate_range(self, handle: int, offset: int, size: int) -> None:
        record = self._get_record(handle)
        if offset < 0:
            raise AllocatorError("invalid_request", "offset must be non-negative")
        if size < 0:
            raise AllocatorError("invalid_request", "size must be non-negative")
        if offset + size > record.nbytes:
            raise AllocatorError(
                "out_of_bounds",
                f"range [{offset}, {offset + size}) exceeds tensor size {record.nbytes}",
            )

    def _walk(self, handle: int, offset: int, size: int):
        record = self._get_record(handle)
        logical = 0
        remaining = size

        for extent in record.extents:
            extent_start = logical
            extent_end = logical + extent.nbytes
            logical = extent_end

            if offset >= extent_end:
                continue

            local_offset = max(0, offset - extent_start)
            count = min(remaining, extent.nbytes - local_offset)
            if count <= 0:
                continue

            yield extent, local_offset, count
            remaining -= count
            offset += count
            if remaining == 0:
                return


def describe_record(record: TensorRecord) -> dict[str, object]:
    return {
        "handle": record.handle,
        "nbytes": record.nbytes,
        "shape": list(record.shape),
        "dtype": record.dtype,
        "usage": record.usage,
        "layout": record.layout,
        "extent_count": len(record.extents),
    }


def fake_allocator(total_bytes: int, slab_bytes: int = 32 * MIB) -> TensorAllocator:
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if slab_bytes <= 0:
        raise ValueError("slab_bytes must be positive")

    slabs = []
    remaining = total_bytes
    base_address = 0x1000_0000
    while remaining:
        size = min(slab_bytes, remaining)
        slabs.append(FakeSlab(size=size, physical_address=base_address))
        base_address += size + 0x1000
        remaining -= size
    return TensorAllocator(slabs)


def pynq_allocator(total_bytes: int, slab_bytes: int = 32 * MIB) -> TensorAllocator:
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if slab_bytes <= 0:
        raise ValueError("slab_bytes must be positive")

    slabs = []
    remaining = total_bytes
    try:
        while remaining:
            size = min(slab_bytes, remaining)
            slabs.append(PynqSlab(size=size))
            remaining -= size
        return TensorAllocator(slabs)
    except Exception:
        for slab in slabs:
            slab.close()
        raise

