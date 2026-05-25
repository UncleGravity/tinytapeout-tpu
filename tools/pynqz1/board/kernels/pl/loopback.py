"""PL-accelerated COPY via the AXI DMA loopback bitstream.

Semantically equivalent to ``board.kernels.ps.native.Copy`` — same wire
op (``GOP_COPY``), same arguments. The only difference is that bytes
traverse the PL fabric (DMA out → axis_loopback → DMA in) instead of a
host-side memcpy.

This exists primarily as a proof-of-integration: once a vanilla COPY
graph passes through the daemon with this kernel registered, every
plumbing layer (overlay loader, AXI control, AXI HP DMA, cache
maintenance, allocator.slab_view, kernel registry override) is known-
good. Subsequent compute kernels (W1A8 matmul etc.) only have to worry
about their math.
"""

from __future__ import annotations

from typing import Any

from board.kernels.registry import KernelRegistry
from board.memory.allocator import TensorAllocator
from board.profiling.timer import Timer
from proto.ops import F_DST, F_DST_OFFSET, F_NBYTES, F_SRC, F_SRC_OFFSET, GOP_COPY


class PLLoopback:
    name = GOP_COPY

    def __init__(self, overlay):
        # PYNQ exposes the AXI DMA IP block by the name from the .hwh.
        # The build.tcl in fpga/benchmarks/dma_loopback names it
        # ``axi_dma_0`` — adjust here if a future bitstream renames it.
        self._dma = overlay.axi_dma_0

    def run(self, allocator: TensorAllocator, op: dict[str, Any], timer: Timer) -> None:
        nbytes = int(op[F_NBYTES])
        src_off = int(op.get(F_SRC_OFFSET, 0))
        dst_off = int(op.get(F_DST_OFFSET, 0))

        # Resolve handles to underlying slabs. PYNQ DMA accepts a sliced
        # CMA buffer directly; we need the actual buffer + absolute offset.
        src_slab, src_abs, _ = allocator.slab_view(int(op[F_SRC]), src_off, nbytes)
        dst_slab, dst_abs, _ = allocator.slab_view(int(op[F_DST]), dst_off, nbytes)

        with timer.section("flush"):
            src_slab.flush_range(src_abs, nbytes)

        with timer.section("compute"):
            src_view = src_slab.pynq_buffer[src_abs : src_abs + nbytes]
            dst_view = dst_slab.pynq_buffer[dst_abs : dst_abs + nbytes]
            self._dma.sendchannel.transfer(src_view)
            self._dma.recvchannel.transfer(dst_view)
            self._dma.sendchannel.wait()
            self._dma.recvchannel.wait()

        with timer.section("invalidate"):
            dst_slab.invalidate_range(dst_abs, nbytes)

        timer.add("bytes_read", nbytes)
        timer.add("bytes_written", nbytes)


def register_all(registry: KernelRegistry, overlay) -> None:
    """Register every PL kernel against an already-loaded overlay.

    Names that match existing registrations (e.g. ``GOP_COPY``) replace
    them — that's how the PS→PL swap-point works.
    """
    registry.register(PLLoopback(overlay))
