"""Tensor allocator with multi-slab spanning.

A tensor that fits in one slab gets one extent. A tensor too large for
any single slab gets multiple extents that together cover ``nbytes``.
This is required on PYNQ-Z1: CMA fragments at runtime (the DRM driver
shares the pool) so the largest contiguous CMA buffer we can hand the
allocator is ~32 MiB even when much more is "free" overall, while a
ggml weight buffer can be ~230 MiB.

PS kernels do not care about extents — they go through ``read``/``write``
which walks them transparently. PL kernels that want zero-copy DMA can
ask for ``extents(handle)`` and program one descriptor per extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from board.memory.slabs import FakeSlab, PynqSlab, fake_slabs, pynq_slabs


class AllocatorError(RuntimeError):
    """Allocator failure carrying a stable error code for RPC use."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Extent:
    """One contiguous range inside a single slab."""

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

    @property
    def is_contiguous(self) -> bool:
        return len(self.extents) <= 1


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


class TensorAllocator:
    def __init__(
        self,
        slabs: Iterable[FakeSlab | PynqSlab],
        default_alignment: int = 64,
    ):
        self._slabs = list(slabs)
        if not self._slabs:
            raise ValueError("at least one slab is required")

        # Each slab tracks its own free list of (offset, size), sorted by
        # offset and kept coalesced.
        self._free: list[list[tuple[int, int]]] = [
            [(0, slab.size)] for slab in self._slabs
        ]
        self._records: dict[int, TensorRecord] = {}
        self._next_handle = 1
        self.default_alignment = default_alignment

    # -- accounting --------------------------------------------------------

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

    # -- public API --------------------------------------------------------

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
        extents: tuple[Extent, ...]
        if nbytes == 0:
            extents = ()
        else:
            placed = self._place_single(nbytes, align) or self._place_multi(nbytes, align)
            if placed is None:
                raise AllocatorError(
                    "out_of_memory",
                    f"cannot allocate {nbytes} bytes (free {self.free_bytes})",
                )
            extents = placed

        handle = self._next_handle
        self._next_handle += 1
        record = TensorRecord(
            handle=handle,
            nbytes=nbytes,
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype,
            usage=usage,
            layout=layout,
            extents=extents,
        )
        self._records[handle] = record
        return record

    def free(self, handle: int) -> TensorRecord:
        record = self._get_record(handle)
        del self._records[handle]
        for extent in record.extents:
            self._release(extent.slab_index, extent.offset, extent.nbytes)
        return record

    def write(self, handle: int, offset: int, data: bytes | bytearray | memoryview) -> None:
        view = memoryview(data)
        self._validate_range(handle, offset, len(view))
        src_offset = 0
        for extent, local_offset, count in self._walk(handle, offset, len(view)):
            self._slabs[extent.slab_index].write(
                extent.offset + local_offset,
                view[src_offset : src_offset + count],
            )
            src_offset += count

    def read(self, handle: int, offset: int, size: int) -> bytes:
        self._validate_range(handle, offset, size)
        parts: list[bytes] = []
        for extent, local_offset, count in self._walk(handle, offset, size):
            parts.append(self._slabs[extent.slab_index].read(extent.offset + local_offset, count))
        return b"".join(parts)

    def describe(self, handle: int) -> dict[str, object]:
        return describe_record(self._get_record(handle))

    def extents(self, handle: int) -> tuple[Extent, ...]:
        """For zero-copy DMA: every (slab_index, offset, nbytes) in order."""
        return self._get_record(handle).extents

    def physical(self, handle: int, offset: int = 0) -> int:
        """Physical address of ``offset`` bytes into the tensor.

        Raises if the requested range crosses an extent boundary — callers
        that need to span must use ``extents()`` and program one DMA
        descriptor per extent.
        """
        record = self._get_record(handle)
        if offset < 0 or offset > record.nbytes:
            raise AllocatorError(
                "out_of_bounds", f"offset {offset} outside tensor {handle}"
            )
        position = 0
        for extent in record.extents:
            end = position + extent.nbytes
            if offset < end:
                return self._slabs[extent.slab_index].physical_address + extent.offset + (offset - position)
            position = end
        # offset == nbytes — one past the end of a zero-byte tensor
        return 0

    def close(self) -> None:
        for slab in self._slabs:
            slab.close()

    # -- placement --------------------------------------------------------

    def _place_single(self, nbytes: int, alignment: int) -> tuple[Extent, ...] | None:
        """Try to fit the whole tensor in a single slab."""
        for slab_index, slab_free in enumerate(self._free):
            for block_offset, block_size in slab_free:
                aligned = align_up(block_offset, alignment)
                if aligned + nbytes <= block_offset + block_size:
                    self._take(slab_index, aligned, nbytes)
                    return (Extent(slab_index, aligned, nbytes),)
        return None

    def _place_multi(self, nbytes: int, alignment: int) -> tuple[Extent, ...] | None:
        """Greedy multi-slab fill; rolls back on failure."""
        extents: list[Extent] = []
        remaining = nbytes
        for slab_index in range(len(self._slabs)):
            while remaining > 0:
                taken = self._take_largest(slab_index, remaining, alignment)
                if taken is None:
                    break
                offset, count = taken
                extents.append(Extent(slab_index, offset, count))
                remaining -= count
            if remaining == 0:
                break
        if remaining != 0:
            for extent in extents:
                self._release(extent.slab_index, extent.offset, extent.nbytes)
            return None
        return tuple(extents)

    def _take(self, slab_index: int, offset: int, nbytes: int) -> None:
        """Mark [offset, offset+nbytes) busy. The block must already cover it."""
        slab_free = self._free[slab_index]
        for block_index, (block_offset, block_size) in enumerate(slab_free):
            block_end = block_offset + block_size
            if not (block_offset <= offset and offset + nbytes <= block_end):
                continue
            replacement: list[tuple[int, int]] = []
            if offset > block_offset:
                replacement.append((block_offset, offset - block_offset))
            tail_offset = offset + nbytes
            if tail_offset < block_end:
                replacement.append((tail_offset, block_end - tail_offset))
            slab_free[block_index : block_index + 1] = replacement
            return
        raise AllocatorError("internal_error", "free list lost track of a reservation")

    def _take_largest(
        self, slab_index: int, max_size: int, alignment: int
    ) -> tuple[int, int] | None:
        """Take up to ``max_size`` from the first aligned-fittable block."""
        slab_free = self._free[slab_index]
        for block_index, (offset, size) in enumerate(slab_free):
            aligned = align_up(offset, alignment)
            padding = aligned - offset
            available = size - padding
            if available <= 0:
                continue
            take = min(max_size, available)
            replacement: list[tuple[int, int]] = []
            if padding:
                replacement.append((offset, padding))
            tail_offset = aligned + take
            tail_size = size - padding - take
            if tail_size:
                replacement.append((tail_offset, tail_size))
            slab_free[block_index : block_index + 1] = replacement
            return aligned, take
        return None

    def _release(self, slab_index: int, offset: int, nbytes: int) -> None:
        slab_free = self._free[slab_index]
        slab_free.append((offset, nbytes))
        slab_free.sort()
        merged: list[tuple[int, int]] = []
        for block_offset, block_size in slab_free:
            if not merged:
                merged.append((block_offset, block_size))
                continue
            prev_offset, prev_size = merged[-1]
            prev_end = prev_offset + prev_size
            if block_offset <= prev_end:
                merged[-1] = (prev_offset, max(prev_end, block_offset + block_size) - prev_offset)
            else:
                merged.append((block_offset, block_size))
        self._free[slab_index] = merged

    # -- internal helpers -------------------------------------------------

    def _get_record(self, handle: int) -> TensorRecord:
        try:
            return self._records[int(handle)]
        except (KeyError, ValueError):
            raise AllocatorError(
                "unknown_tensor", f"unknown tensor handle {handle}"
            ) from None

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

    def _walk(self, handle: int, offset: int, size: int) -> Iterator[tuple[Extent, int, int]]:
        record = self._get_record(handle)
        cursor = 0
        remaining = size
        for extent in record.extents:
            extent_end = cursor + extent.nbytes
            if offset >= extent_end:
                cursor = extent_end
                continue
            local_offset = max(0, offset - cursor)
            count = min(remaining, extent.nbytes - local_offset)
            if count <= 0:
                cursor = extent_end
                continue
            yield extent, local_offset, count
            remaining -= count
            offset += count
            cursor = extent_end
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


def fake_allocator(total_bytes: int, slab_bytes: int) -> TensorAllocator:
    return TensorAllocator(fake_slabs(total_bytes, slab_bytes))


def pynq_allocator(total_bytes: int, slab_bytes: int) -> TensorAllocator:
    return TensorAllocator(pynq_slabs(total_bytes, slab_bytes))
