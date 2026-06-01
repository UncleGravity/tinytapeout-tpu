"""PL-backed MATMUL_Q1A8 driver for the rowblock bitstream.

Hot path: per-column quantize + merge_acts(packed_weights, quants, scales)
+ DMA. The merge_acts step is a memcpy walk over weights that the host
already pre-packed into AXIS rowblock layout at upload time. No bit
shuffles in the hot path; the legacy per-matmul stream pack that used to
dominate matmul wall time (~85%) is gone.
"""

from __future__ import annotations

import ctypes
import math
import struct
from typing import Any

from board.kernels.ps.native import load_lib
from board.memory.allocator import AllocatorError, TensorAllocator
from board.profiling.timer import Timer
from proto import q1a8_layout
from proto.ops import (
    F_ACTS,
    F_ACTS_OFFSET,
    F_COLS,
    F_DST,
    F_DST_OFFSET,
    F_K,
    F_ROWS,
    F_WEIGHTS,
    F_WEIGHTS_OFFSET,
    GOP_MATMUL_Q1A8,
    Q1_BLOCK,
    Q1_BLOCK_BYTES,
    Q8_BLOCK,
)

F32_BYTES = 4
ROWS_PER_BLOCK = q1a8_layout.ROWS_PER_BLOCK

REG_ID = 0x00
REG_VERSION = 0x04
REG_CTRL = 0x08
REG_STATUS = 0x0C
REG_NUM_Q1_BLOCKS = 0x10
REG_NUM_ROWBLOCKS = 0x14
REG_CYCLES = 0x18
REG_ROWS = 0x1C

CTRL_START = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID = 0xB05A_2000
EXPECTED_VERSION = 4  # v4 = dual-stream (acts via S_AXIS_ACTS, weights via S_AXIS)
# Used by the C runner for ALL DMA + kernel waits. Each MMIO poll is ~100ns
# on Cortex-A9, so 10M polls ≈ 1s — plenty for an 8 MiB DMA chunk (~11ms)
# but bounded so a hung kernel surfaces as RuntimeError rather than hang.
POLL_LIMIT = 10_000_000


def _required(op: dict[str, Any], key: str) -> int:
    if key not in op:
        raise AllocatorError("invalid_request", f"missing {key}")
    value = int(op[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


def _optional_int(op: dict[str, Any], key: str, default: int = 0) -> int:
    if key not in op:
        return default
    value = int(op[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


# -- ctypes bindings ------------------------------------------------------
#
# The PL driver shares libbonsai_ps.so with the PS kernels. The two entry
# points used here are the wire-stream packer and the fp16-bits Q8_0
# quantizer; both replace per-rowblock Python loops that dominated the
# previous driver's per-token wall time.


def _bind_native(lib: ctypes.CDLL):
    quantize = lib.bonsai_quantize_q8_0_pl
    quantize.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    quantize.restype = ctypes.c_int

    # v4 acts packer: one column's quants + scales → S_AXIS_ACTS wire bytes.
    # Pure memcpy + write_u64; the v3 merge function is no longer in the hot
    # path (the kernel BRAMs acts at start and broadcasts to all rowblocks).
    pack_acts = lib.bonsai_q1a8_pack_acts
    pack_acts.argtypes = [
        ctypes.POINTER(ctypes.c_int8),   # act_quants
        ctypes.POINTER(ctypes.c_uint16), # act_scale_bits
        ctypes.c_uint32,                 # k
        ctypes.POINTER(ctypes.c_uint8),  # out
    ]
    pack_acts.restype = ctypes.c_int

    # Single-call orchestration: kernel + DMA reg pokes + wait loops in C
    # (q1a8_runner.c). Cuts ~10 Python sections out of the hot per-matmul
    # path, killing the ctypes/PYNQ wrapper overhead that dominated v4 wall
    # time. Caller (Python) still handles cache flush/invalidate and slab
    # pointer resolution.
    run_chunk = lib.bonsai_q1a8_run_matmul_chunk
    run_chunk.argtypes = [
        ctypes.c_void_p,                 # kernel_regs (volatile uint32_t *)
        ctypes.c_void_p,                 # dma_w_regs
        ctypes.c_void_p,                 # dma_a_regs
        ctypes.c_uint32,                 # weights_phys_addr
        ctypes.c_uint32,                 # weights_nbytes
        ctypes.c_uint32,                 # acts_phys_addr
        ctypes.c_uint32,                 # acts_nbytes
        ctypes.c_uint32,                 # result_phys_addr
        ctypes.c_uint32,                 # result_nbytes
        ctypes.c_uint32,                 # num_q1_blocks
        ctypes.c_uint32,                 # num_rowblocks
        ctypes.c_uint32,                 # poll_limit
        ctypes.POINTER(ctypes.c_uint32), # out_cycles
    ]
    run_chunk.restype = ctypes.c_int

    # Pipelining split of run_chunk: start kicks the DMAs + kernel and returns
    # immediately; wait polls them to completion. Serial start()+wait() is
    # identical to run_chunk(); the scheduler interleaves PS work between them.
    start_chunk = lib.bonsai_q1a8_start_matmul_chunk
    start_chunk.argtypes = [
        ctypes.c_void_p,                 # kernel_regs
        ctypes.c_void_p,                 # dma_w_regs
        ctypes.c_void_p,                 # dma_a_regs
        ctypes.c_uint32,                 # weights_phys_addr
        ctypes.c_uint32,                 # weights_nbytes
        ctypes.c_uint32,                 # acts_phys_addr
        ctypes.c_uint32,                 # acts_nbytes
        ctypes.c_uint32,                 # result_phys_addr
        ctypes.c_uint32,                 # result_nbytes
        ctypes.c_uint32,                 # num_q1_blocks
        ctypes.c_uint32,                 # num_rowblocks
    ]
    start_chunk.restype = ctypes.c_int

    wait_chunk = lib.bonsai_q1a8_wait_matmul_chunk
    wait_chunk.argtypes = [
        ctypes.c_void_p,                 # kernel_regs
        ctypes.c_void_p,                 # dma_w_regs
        ctypes.c_void_p,                 # dma_a_regs
        ctypes.c_uint32,                 # poll_limit
        ctypes.POINTER(ctypes.c_uint32), # out_cycles
    ]
    wait_chunk.restype = ctypes.c_int
    return quantize, pack_acts, run_chunk, start_chunk, wait_chunk


# -- legacy Python helpers (kept for tests & golden comparisons) ----------


def _lround_like_native(value: float) -> int:
    if value >= 0.0:
        return int(value + 0.5)
    return int(value - 0.5)


def _fp16_float_to_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _quantize_q8_0(values: tuple[float, ...]) -> tuple[list[int], list[int]]:
    """Python reference for Q8_0 quantization. Matches the C ``bonsai_quantize_q8_0_pl``."""
    if len(values) % Q8_BLOCK != 0:
        raise ValueError("Q8_0 input length must be a multiple of Q8_BLOCK")

    quants = [0] * len(values)
    scale_bits = [0] * (len(values) // Q8_BLOCK)

    for block_index, block_start in enumerate(range(0, len(values), Q8_BLOCK)):
        block = values[block_start : block_start + Q8_BLOCK]
        amax = 0.0
        for value in block:
            if math.isfinite(value):
                amax = max(amax, abs(value))
        if amax == 0.0:
            continue

        scale = amax / 127.0
        scale_bits[block_index] = _fp16_float_to_bits(scale)
        inv_scale = 1.0 / scale
        for local_index, value in enumerate(block):
            if not math.isfinite(value):
                continue
            quant = _lround_like_native(value * inv_scale)
            quants[block_start + local_index] = min(127, max(-128, quant))

    return quants, scale_bits


def _q1_bytes_per_rowblock(rows_per_block: int = ROWS_PER_BLOCK) -> int:
    scale_beats = (rows_per_block + 3) // 4
    wbits_beats = (rows_per_block + 1) // 2
    return (scale_beats + 4 * (5 + wbits_beats)) * 8


def _rowblock_nbytes(k: int, rows_per_block: int = ROWS_PER_BLOCK) -> int:
    return (k // Q1_BLOCK) * _q1_bytes_per_rowblock(rows_per_block)


def _pack_rowblock_into(
    out: bytearray,
    weights: bytes,
    row_start: int,
    row_count: int,
    weight_row_bytes: int,
    act_quants: list[int],
    act_scale_bits: list[int],
    k: int,
    rows_per_block: int = ROWS_PER_BLOCK,
) -> None:
    """Python reference for packing one rowblock. Used by tests; the runtime path
    uses the C implementation via ``bonsai_pack_matmul_q1a8_stream``."""
    expected_len = _rowblock_nbytes(k, rows_per_block)
    if len(out) != expected_len:
        raise ValueError(f"packed buffer must be {expected_len} bytes")
    if row_count < 1 or row_count > rows_per_block:
        raise ValueError("row_count outside rowblock bounds")

    scale_beats = (rows_per_block + 3) // 4
    wbits_beats = (rows_per_block + 1) // 2
    blocks_per_row = k // Q1_BLOCK
    cursor = 0

    for q1_index in range(blocks_per_row):
        for beat in range(scale_beats):
            word = 0
            for local in range(4):
                lane = beat * 4 + local
                scale = 0
                if lane < row_count:
                    block_offset = (
                        (row_start + lane) * weight_row_bytes
                        + q1_index * Q1_BLOCK_BYTES
                    )
                    scale = struct.unpack_from("<H", weights, block_offset)[0]
                word |= scale << (local * 16)
            struct.pack_into("<Q", out, cursor, word)
            cursor += 8

        for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
            q8_base = q1_index * Q1_BLOCK + q8_local
            out[cursor : cursor + Q8_BLOCK] = bytes(
                quant & 0xFF for quant in act_quants[q8_base : q8_base + Q8_BLOCK]
            )
            cursor += Q8_BLOCK

            struct.pack_into("<Q", out, cursor, act_scale_bits[q8_base // Q8_BLOCK])
            cursor += 8

            for beat in range(wbits_beats):
                word = 0
                for local in range(2):
                    lane = beat * 2 + local
                    bits = 0
                    if lane < row_count:
                        block_offset = (
                            (row_start + lane) * weight_row_bytes
                            + q1_index * Q1_BLOCK_BYTES
                        )
                        bits_offset = block_offset + 2 + q8_local // 8
                        bits = int.from_bytes(weights[bits_offset : bits_offset + 4], "little")
                    word |= bits << (local * 32)
                struct.pack_into("<Q", out, cursor, word)
                cursor += 8


# -- runtime driver -------------------------------------------------------


class PLMatmulQ1A8:
    name = GOP_MATMUL_Q1A8
    backend = "pl"

    def __init__(self, overlay):
        # v4 bitstream has two DMAs: weights+results on axi_dma_0, acts on
        # axi_dma_1 (MM2S only).
        self._overlay = overlay
        self._dma_w = overlay.axi_dma_0
        self._dma_a = overlay.axi_dma_1
        self._kernel = overlay.q1a8_kernel_top_0
        self._acts_buf = None
        self._acts_buf_size = 0
        self._weights_dma_buf = None     # CMA scratch for multi-extent chunks
        self._weights_dma_buf_size = 0
        self._result_buf = None
        self._result_buf_size = 0
        self._np = None
        self._rows_per_block = ROWS_PER_BLOCK
        self._lib = None
        self._c_quantize = None
        self._c_pack_acts = None
        self._c_run_chunk = None
        self._c_start_chunk = None
        self._c_wait_chunk = None
        # MMIO pointers passed to the C runner. Resolved lazily on first
        # run() (overlay is fully wired by then) since some tests construct
        # PLMatmulQ1A8 against a fake overlay with no real ip_dict.
        self._kernel_regs_ptr = None
        self._dma_w_regs_ptr = None
        self._dma_a_regs_ptr = None
        self._mmio_refs: list = []  # keep MMIO objects alive
        # Lazy CMA scratch buffers for the rare case where a tensor spans
        # slab extents: allocator.slab_pointer refuses to hand out a single
        # pointer there, so we copy the bytes into a contiguous CMA scratch
        # and pass the scratch's pointer to C.
        self._weights_scratch = None
        self._weights_scratch_size = 0
        self._acts_in_scratch = None
        self._acts_in_scratch_size = 0

    def _resolve_slab_pointer(
        self,
        allocator: TensorAllocator,
        handle: int,
        offset: int,
        nbytes: int,
        kind: str,
    ) -> int:
        """Return a CMA address covering ``nbytes`` of tensor data.

        Fast path: the range lives in a single slab extent, so we return the
        slab's user-space VA directly (zero copy). Fallback: the range spans
        extents, so we copy it through a persistent CMA scratch reserved for
        this op channel.
        """
        try:
            return allocator.slab_pointer(handle, offset, nbytes)
        except AllocatorError as exc:
            if exc.code != "multi_extent":
                raise
            return self._copy_to_cma_scratch(allocator, handle, offset, nbytes, kind)

    def _copy_to_cma_scratch(
        self,
        allocator: TensorAllocator,
        handle: int,
        offset: int,
        nbytes: int,
        kind: str,
    ) -> int:
        import ctypes as _ct

        import numpy as np
        from pynq import allocate

        if kind == "weights":
            buf = self._weights_scratch
            size = self._weights_scratch_size
        else:
            buf = self._acts_in_scratch
            size = self._acts_in_scratch_size

        if buf is None or size < nbytes:
            # Orphan the old buffer — Python GC will release it via PynqBuffer
            # finalization. Calling buf.freebuffer() directly has been
            # observed to raise AttributeError ('PynqBuffer' object has no
            # attribute 'bo') in some lifecycle states; the GC path goes
            # through the same release without touching .bo from our code.
            buf = allocate(shape=(nbytes,), dtype=np.uint8)
            size = nbytes

        if kind == "weights":
            self._weights_scratch = buf
            self._weights_scratch_size = size
        else:
            self._acts_in_scratch = buf
            self._acts_in_scratch_size = size

        src = allocator.read(handle, offset, nbytes)
        src_arr = np.frombuffer(src, dtype=np.uint8)
        # Use ctypes.memmove instead of buf[:nbytes] = src_arr. PynqBuffer's
        # __setitem__ has a cache-coherence hook that touches .bo, which is
        # missing for some slice/view states and raises AttributeError. A
        # raw memmove bypasses pynq's hook entirely; no DMA is involved here
        # so no explicit flush is needed (the C reader uses CPU loads and
        # sees the just-written cache lines).
        _ct.memmove(
            buf.ctypes.data_as(_ct.c_void_p),
            src_arr.ctypes.data_as(_ct.c_void_p),
            nbytes,
        )
        return buf.ctypes.data_as(_ct.c_void_p).value

    def _ensure_native(self) -> None:
        # Lazy: the host-side registration test constructs PLMatmulQ1A8 without
        # the on-board libbonsai_ps.so being available, so loading is deferred
        # until the first run().
        if self._c_quantize is not None:
            return
        self._lib = load_lib()
        (self._c_quantize, self._c_pack_acts, self._c_run_chunk,
         self._c_start_chunk, self._c_wait_chunk) = _bind_native(self._lib)

    def _ensure_mmio_pointers(self) -> None:
        """Resolve MMIO base addresses for the C runner.

        Called lazily on first run() — the overlay's ip_dict isn't fully
        populated until the bitstream is actually loaded, which doesn't
        happen in the host-side registration tests.
        """
        if self._kernel_regs_ptr is not None:
            return
        from pynq import MMIO

        # Kernel: use the kernel object's existing MMIO so it's the same
        # view Python uses for ID/version checks. PYNQ exposes it as .mmio
        # on the DefaultIP-derived kernel object.
        kernel_mmio = self._kernel.mmio
        self._mmio_refs.append(kernel_mmio)
        self._kernel_regs_ptr = kernel_mmio.array.ctypes.data

        # DMAs: open our own MMIO views — PYNQ's DMA Python object wraps
        # the registers in classes we can't directly hand to C.
        ip_dict = self._overlay.ip_dict
        for name, attr in (("axi_dma_0", "_dma_w_regs_ptr"),
                           ("axi_dma_1", "_dma_a_regs_ptr")):
            info = ip_dict[name]
            mmio = MMIO(info["phys_addr"], info["addr_range"])
            self._mmio_refs.append(mmio)
            setattr(self, attr, mmio.array.ctypes.data)

    def run(self, allocator: TensorAllocator, op: dict[str, Any], timer: Timer) -> None:
        rows = _required(op, F_ROWS)
        cols = _required(op, F_COLS)
        k = _required(op, F_K)
        if rows == 0 or cols == 0:
            raise AllocatorError("invalid_request", "rows and cols must be positive")
        if k == 0 or k % Q1_BLOCK != 0:
            raise AllocatorError(
                "invalid_request",
                f"{GOP_MATMUL_Q1A8} k must be a positive multiple of {Q1_BLOCK}",
            )

        timer.add("rows", rows)
        timer.add("cols", cols)
        timer.add("k", k)

        self._ensure_native()
        self._ensure_mmio_pointers()

        blocks_per_row = k // Q1_BLOCK
        packed_bytes_per_rb = q1a8_layout.packed_bytes_per_rowblock(k)
        weight_nbytes = q1a8_layout.packed_nbytes(rows, k)
        acts_stream_nbytes = q1a8_layout.acts_stream_nbytes(k)
        act_nbytes = cols * k * F32_BYTES
        dst_nbytes = rows * cols * F32_BYTES

        weights_handle = _required(op, F_WEIGHTS)
        weights_base_offset = _optional_int(op, F_WEIGHTS_OFFSET)
        with timer.section("read"):
            acts_addr = self._resolve_slab_pointer(
                allocator,
                _required(op, F_ACTS),
                _optional_int(op, F_ACTS_OFFSET),
                act_nbytes,
                "acts",
            )
        timer.add("bytes_read", weight_nbytes + act_nbytes)

        self._check_kernel_id()
        rows_per_block = self._read_rows_per_block()
        rowblocks_per_col = (rows + rows_per_block - 1) // rows_per_block

        # Chunk by weight bytes. The kernel can in principle process all
        # rowblocks in one start, but lm_head (~44 MiB packed) spans multiple
        # CMA slab extents — and DMA needs a single contiguous pointer per
        # transfer, so we chunk to fit a slab extent and the result buffer.
        max_weights_nbytes = 8 * 1024 * 1024
        max_rowblocks_per_chunk = min(
            rowblocks_per_col,
            max(1, max_weights_nbytes // packed_bytes_per_rb),
        )
        chunk_weights_nbytes_cap = max_rowblocks_per_chunk * packed_bytes_per_rb
        chunk_result_nbytes = max_rowblocks_per_chunk * rows_per_block * F32_BYTES

        import numpy as np

        self._np = np
        self._ensure_buffers(acts_stream_nbytes, chunk_weights_nbytes_cap,
                             chunk_result_nbytes)

        act_quants = np.empty(k, dtype=np.int8)
        act_scale_bits = np.empty(k // Q8_BLOCK, dtype=np.uint16)

        quants_ptr = act_quants.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        scales_ptr = act_scale_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        acts_buf_ptr = self._acts_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        dst_handle = _required(op, F_DST)
        dst_base_offset = _optional_int(op, F_DST_OFFSET)
        total_cycles = 0
        # dst slabs the kernel wrote into via direct S2MM; invalidated once
        # at the end so downstream CPU reads see the DMA'd data. Keyed by
        # id(slab) → [slab, lo, hi] so a future range-invalidate is exact.
        dst_dirty: dict[int, list] = {}

        for col in range(cols):
            col_acts_offset = col * k * F32_BYTES

            with timer.section("quantize"):
                acts_in_ptr = ctypes.cast(
                    acts_addr + col_acts_offset,
                    ctypes.POINTER(ctypes.c_float),
                )
                rc = self._c_quantize(acts_in_ptr, ctypes.c_uint32(k),
                                      quants_ptr, scales_ptr)
                if rc != 0:
                    raise RuntimeError(f"bonsai_quantize_q8_0_pl rc={rc}")

            # One pack per column: small (k * 1.25 bytes), pure memcpy.
            with timer.section("pack_acts"):
                rc = self._c_pack_acts(
                    quants_ptr, scales_ptr, ctypes.c_uint32(k), acts_buf_ptr,
                )
                if rc != 0:
                    raise RuntimeError(f"bonsai_q1a8_pack_acts rc={rc}")
                self._acts_buf.flush()

            row_chunk_start = 0
            rowblock_chunk_start = 0
            while row_chunk_start < rows:
                # Size the chunk so its weights stay inside one slab extent:
                # then _resolve_weights_phys hits the fast path (a physical
                # address) instead of copying the chunk through CPU scratch.
                remaining_rowblocks = rowblocks_per_col - rowblock_chunk_start
                chunk_weight_offset = (
                    weights_base_offset
                    + rowblock_chunk_start * packed_bytes_per_rb
                )
                ext_remaining = self._extent_remaining(
                    allocator, weights_handle, chunk_weight_offset)
                rowblocks_to_boundary = ext_remaining // packed_bytes_per_rb
                chunk_rowblocks = min(max_rowblocks_per_chunk, remaining_rowblocks)
                if rowblocks_to_boundary >= 1:
                    chunk_rowblocks = min(chunk_rowblocks, rowblocks_to_boundary)
                else:
                    # A single rowblock straddles an extent boundary (rare):
                    # take just it; _resolve_weights_phys copies it via scratch.
                    chunk_rowblocks = 1

                chunk_rows = min(chunk_rowblocks * rows_per_block, rows - row_chunk_start)
                chunk_weights_nbytes = chunk_rowblocks * packed_bytes_per_rb
                # The kernel emits rows_per_block fp32 per rowblock; the last
                # rowblock of an unaligned tensor over-emits past chunk_rows.
                padded_result_nbytes = chunk_rowblocks * rows_per_block * F32_BYTES
                actual_result_nbytes = chunk_rows * F32_BYTES
                dst_chunk_offset = (
                    dst_base_offset + (col * rows + row_chunk_start) * F32_BYTES
                )

                # Fast path: stream the kernel result straight into the dst
                # slab (no result_buf → bytearray → allocator.write detour).
                # Needs the full padded extent to land contiguously in one
                # slab, so skip it for a partial trailing rowblock.
                dst_slab = None
                if padded_result_nbytes == actual_result_nbytes:
                    try:
                        dst_slab, dst_abs, _ = allocator.slab_view(
                            dst_handle, dst_chunk_offset, padded_result_nbytes)
                        result_phys = dst_slab.pynq_buffer.physical_address + dst_abs
                    except AllocatorError as exc:
                        if exc.code != "multi_extent":
                            raise
                        dst_slab = None
                if dst_slab is None:
                    result_phys = self._result_buf.physical_address

                with timer.section("compute"):
                    cycles = self._run_matmul_dual_stream(
                        allocator,
                        weights_handle,
                        chunk_weight_offset,
                        chunk_weights_nbytes,
                        acts_stream_nbytes,
                        result_phys,
                        padded_result_nbytes,
                        blocks_per_row,
                        chunk_rowblocks,
                        timer,
                    )
                total_cycles += cycles

                if dst_slab is not None:
                    entry = dst_dirty.get(id(dst_slab))
                    if entry is None:
                        dst_dirty[id(dst_slab)] = [
                            dst_slab, dst_abs, dst_abs + actual_result_nbytes]
                    else:
                        entry[1] = min(entry[1], dst_abs)
                        entry[2] = max(entry[2], dst_abs + actual_result_nbytes)
                else:
                    # Fallback: kernel wrote result_buf; copy just this chunk
                    # back into its dst range.
                    with timer.section("result_invalidate"):
                        self._result_buf[:padded_result_nbytes].invalidate()
                    with timer.section("result_copy"):
                        result_view = self._np.frombuffer(
                            self._result_buf, dtype=self._np.uint8,
                            count=actual_result_nbytes,
                        )
                        chunk_bytes = result_view.tobytes()
                    with timer.section("write"):
                        allocator.write(dst_handle, dst_chunk_offset, chunk_bytes)

                row_chunk_start += chunk_rows
                rowblock_chunk_start += chunk_rowblocks

        # Drop stale CPU cache lines for the directly-DMA'd dst ranges so the
        # next op reads the kernel's output, not pre-DMA contents.
        with timer.section("result_invalidate"):
            for dst_slab, lo, hi in dst_dirty.values():
                dst_slab.invalidate_range(lo, hi - lo)
        timer.add("bytes_written", dst_nbytes)
        timer.add("matmul_cols", cols)
        timer.add("rowblocks", cols * rowblocks_per_col)
        # Per-matmul: weight bytes DMA'd once; acts stream DMA'd once per col.
        timer.add("dma_bytes_read",
                  cols * (weight_nbytes + acts_stream_nbytes))
        timer.add(
            "dma_bytes_written",
            cols * rowblocks_per_col * rows_per_block * F32_BYTES,
        )
        timer.add("kernel_cycles", total_cycles)

    def _check_kernel_id(self) -> None:
        got_id = self._kernel.read(REG_ID)
        got_version = self._kernel.read(REG_VERSION)
        if isinstance(got_id, int) and got_id != EXPECTED_ID:
            raise RuntimeError(f"q1a8 rowblock kernel ID mismatch: got 0x{got_id:08x}")
        if isinstance(got_version, int) and got_version != EXPECTED_VERSION:
            raise RuntimeError(
                f"q1a8 kernel version mismatch: bitstream reports v{got_version}, "
                f"driver expects v{EXPECTED_VERSION}. The dual-stream driver "
                f"requires a v4 bitstream — rebuild via "
                f"`cd fpga/bitstreams/matmul_q1a8 && ./build.sh --install`."
            )

    def _read_rows_per_block(self) -> int:
        rows_per_block = int(self._kernel.read(REG_ROWS))
        if rows_per_block != ROWS_PER_BLOCK:
            raise RuntimeError(f"q1a8 rowblock lanes mismatch: got {rows_per_block}")
        self._rows_per_block = rows_per_block
        return rows_per_block

    def _ensure_buffers(
        self,
        acts_nbytes: int,
        weights_dma_nbytes: int,
        result_nbytes: int,
    ) -> None:
        """Lazy CMA buffers used by the DMA path.

        acts_nbytes:      v4 acts wire stream (~2.5 KB per column at k=2048)
        weights_dma_nbytes: scratch for the rare multi-extent weight chunk;
                          the fast path DMAs directly out of the weight slab.
        result_nbytes:    per-chunk result fp32 buffer.
        """
        from pynq import allocate

        if self._acts_buf is None or self._acts_buf_size < acts_nbytes:
            if self._acts_buf is not None:
                self._acts_buf.freebuffer()
            self._acts_buf = allocate(shape=(acts_nbytes,), dtype=self._np.uint8)
            self._acts_buf_size = acts_nbytes

        if (self._weights_dma_buf is None
                or self._weights_dma_buf_size < weights_dma_nbytes):
            if self._weights_dma_buf is not None:
                self._weights_dma_buf.freebuffer()
            self._weights_dma_buf = allocate(
                shape=(weights_dma_nbytes,), dtype=self._np.uint8)
            self._weights_dma_buf_size = weights_dma_nbytes

        if self._result_buf is None or self._result_buf_size < result_nbytes:
            if self._result_buf is not None:
                self._result_buf.freebuffer()
            self._result_buf = allocate(shape=(result_nbytes,), dtype=self._np.uint8)
            self._result_buf_size = result_nbytes

    def _resolve_weights_phys(
        self,
        allocator: TensorAllocator,
        handle: int,
        offset: int,
        nbytes: int,
    ) -> int:
        """Return the CMA physical address of the weight chunk for DMA.

        Fast path: the chunk fits in one slab extent → physical address is
        slab.pynq_buffer.physical_address + extent.offset. Fallback: copy
        the chunk through self._weights_dma_buf scratch and return its
        physical address.
        """
        try:
            slab, abs_off, _ = allocator.slab_view(handle, offset, nbytes)
            return slab.pynq_buffer.physical_address + abs_off
        except AllocatorError as exc:
            if exc.code != "multi_extent":
                raise
        import ctypes as _ct
        assert self._weights_dma_buf is not None
        if nbytes > self._weights_dma_buf_size:
            raise RuntimeError(
                f"weights multi-extent chunk ({nbytes} B) exceeds scratch "
                f"({self._weights_dma_buf_size} B)")
        src = allocator.read(handle, offset, nbytes)
        src_arr = self._np.frombuffer(src, dtype=self._np.uint8)
        _ct.memmove(
            self._weights_dma_buf.ctypes.data_as(_ct.c_void_p),
            src_arr.ctypes.data_as(_ct.c_void_p),
            nbytes,
        )
        self._weights_dma_buf.flush()
        return self._weights_dma_buf.physical_address

    @staticmethod
    def _extent_remaining(
        allocator: TensorAllocator, handle: int, tensor_offset: int,
    ) -> int:
        """Bytes from ``tensor_offset`` to the end of its slab extent.

        Used to size weight chunks so each stays within one extent and the
        DMA can stream straight out of the slab (no scratch copy)."""
        cursor = 0
        for extent in allocator.extents(handle):
            if tensor_offset < cursor + extent.nbytes:
                return cursor + extent.nbytes - tensor_offset
            cursor += extent.nbytes
        return 0

    def _run_matmul_dual_stream(
        self,
        allocator: TensorAllocator,
        weights_handle: int,
        weights_offset: int,
        weights_nbytes: int,
        acts_stream_nbytes: int,
        result_phys: int,
        result_nbytes: int,
        num_q1_blocks: int,
        num_rowblocks: int,
        timer: Timer,
    ) -> int:
        """Drive one column-chunk via the C runner.

        Python does only: resolve the weight physical address and call C.
        ``result_phys`` is where S2MM writes — the dst slab directly on the
        fast path, or the result scratch buffer on the fallback. The caller
        owns result-cache coherence. Everything between (~10 sections in the
        old path) is one bonsai_q1a8_run_matmul_chunk call.
        """
        assert self._acts_buf is not None

        with timer.section("resolve_weights"):
            weights_phys = self._resolve_weights_phys(
                allocator, weights_handle, weights_offset, weights_nbytes)

        # Serial start()+wait() — behaviourally identical to the fused
        # run_chunk(), but split so a future scheduler can run PS work between
        # the two. chunk_start should be ~µs (just MMIO pokes); chunk_wait is
        # where the DMA+kernel wall time lives (= what pipelining will hide).
        out_cycles = ctypes.c_uint32(0)
        with timer.section("run_chunk"):
            with timer.section("chunk_start"):
                rc = self._c_start_chunk(
                    self._kernel_regs_ptr,
                    self._dma_w_regs_ptr,
                    self._dma_a_regs_ptr,
                    ctypes.c_uint32(weights_phys),
                    ctypes.c_uint32(weights_nbytes),
                    ctypes.c_uint32(self._acts_buf.physical_address),
                    ctypes.c_uint32(acts_stream_nbytes),
                    ctypes.c_uint32(result_phys),
                    ctypes.c_uint32(result_nbytes),
                    ctypes.c_uint32(num_q1_blocks),
                    ctypes.c_uint32(num_rowblocks),
                )
            if rc != 0:
                raise RuntimeError(
                    f"bonsai_q1a8_start_matmul_chunk rc={rc} "
                    f"(see q1a8_runner.c for error code mapping)")
            with timer.section("chunk_wait"):
                rc = self._c_wait_chunk(
                    self._kernel_regs_ptr,
                    self._dma_w_regs_ptr,
                    self._dma_a_regs_ptr,
                    ctypes.c_uint32(POLL_LIMIT),
                    ctypes.byref(out_cycles),
                )
            if rc != 0:
                raise RuntimeError(
                    f"bonsai_q1a8_wait_matmul_chunk rc={rc} "
                    f"(see q1a8_runner.c for error code mapping)")

        return int(out_cycles.value)
