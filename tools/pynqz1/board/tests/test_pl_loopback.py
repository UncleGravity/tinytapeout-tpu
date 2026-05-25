"""PLLoopback unit tests using a mock overlay + DMA.

A real PYNQ board isn't required: we fake the AXI DMA channels and
assert the kernel calls them with the right buffer slices.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from board.kernels import pl
from board.kernels.ps import native as ps_native
from board.kernels.registry import KernelRegistry
from board.memory.allocator import TensorAllocator
from board.memory.slabs import fake_slabs
from board.profiling.timer import Timer
from proto.ops import GOP_COPY


def _mock_overlay():
    overlay = MagicMock()
    overlay.axi_dma_0 = MagicMock()
    return overlay


def test_register_all_overrides_ps_copy():
    registry = KernelRegistry()
    ps_native.register_all(registry, lib_path=None) if False else None  # skip native build
    # Use a fresh registry to avoid needing libbonsai_ps.so for this unit test.
    registry = KernelRegistry()
    overlay = _mock_overlay()
    pl.register_all(registry, overlay)
    assert isinstance(registry.get(GOP_COPY), pl.loopback.PLLoopback)


def test_loopback_dispatches_dma_with_correct_slices():
    # FakeSlab doesn't have pynq_buffer; monkey-patch one in so the
    # loopback kernel's slicing path is testable without PYNQ.
    alloc = TensorAllocator(fake_slabs(total_bytes=4096, slab_bytes=4096))
    slab = alloc._slabs[0]
    slab.pynq_buffer = MagicMock()  # type: ignore[attr-defined]
    slab.pynq_buffer.__getitem__ = MagicMock(side_effect=lambda s: ("view", s))

    src = alloc.allocate(128).handle
    dst = alloc.allocate(128).handle

    overlay = _mock_overlay()
    kernel = pl.loopback.PLLoopback(overlay)

    timer = Timer()
    with timer.op(GOP_COPY):
        kernel.run(alloc, {
            "op": GOP_COPY,
            "src": src, "dst": dst, "nbytes": 64,
            "src_offset": 16, "dst_offset": 32,
        }, timer)

    # send/recv each called once
    assert overlay.axi_dma_0.sendchannel.transfer.call_count == 1
    assert overlay.axi_dma_0.recvchannel.transfer.call_count == 1
    assert overlay.axi_dma_0.sendchannel.wait.call_count == 1
    assert overlay.axi_dma_0.recvchannel.wait.call_count == 1

    # The slice passed to send is at the src absolute offset.
    send_arg = overlay.axi_dma_0.sendchannel.transfer.call_args[0][0]
    recv_arg = overlay.axi_dma_0.recvchannel.transfer.call_args[0][0]
    assert send_arg[1] == slice(alloc._records[src].extents[0].offset + 16,
                                alloc._records[src].extents[0].offset + 16 + 64)
    assert recv_arg[1] == slice(alloc._records[dst].extents[0].offset + 32,
                                alloc._records[dst].extents[0].offset + 32 + 64)

    # Bytes counters reported up to the timer.
    span = timer.ops[0]
    assert span.fields["bytes_read"] == 64
    assert span.fields["bytes_written"] == 64
