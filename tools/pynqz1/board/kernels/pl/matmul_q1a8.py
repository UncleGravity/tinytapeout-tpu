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
EXPECTED_VERSION = 3
POLL_LIMIT = 100_000


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

    # Merge: walks packed_weights (one chunk's worth) + per-column acts +
    # scales into the AXIS wire stream. Replaces the old per-matmul pack
    # which read Q1_0 and did the bit shuffles inline.
    merge = lib.bonsai_q1a8_merge_acts
    merge.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # packed_weights
        ctypes.POINTER(ctypes.c_int8),   # act_quants
        ctypes.POINTER(ctypes.c_uint16), # act_scale_bits
        ctypes.c_uint32,                 # rows
        ctypes.c_uint32,                 # k
        ctypes.POINTER(ctypes.c_uint8),  # out_stream
    ]
    merge.restype = ctypes.c_int
    return quantize, merge


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
        self._dma = overlay.axi_dma_0
        self._kernel = overlay.q1a8_kernel_top_0
        self._stream_buf = None
        self._stream_buf_size = 0
        self._result_buf = None
        self._result_buf_size = 0
        self._np = None
        self._rows_per_block = ROWS_PER_BLOCK
        self._lib = None
        self._c_quantize = None
        self._c_merge = None
        # Lazy CMA scratch buffers for the rare case where a tensor spans
        # slab extents: allocator.slab_pointer refuses to hand out a single
        # pointer there, so we copy the bytes into a contiguous CMA scratch
        # and pass the scratch's pointer to C.
        self._weights_scratch = None
        self._weights_scratch_size = 0
        self._acts_scratch = None
        self._acts_scratch_size = 0

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
            buf = self._acts_scratch
            size = self._acts_scratch_size

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
            self._acts_scratch = buf
            self._acts_scratch_size = size

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
        self._c_quantize, self._c_merge = _bind_native(self._lib)

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

        blocks_per_row = k // Q1_BLOCK
        # Packed weight layout (host pre-packed at upload): rowblock-major,
        # PACKED_PER_Q1_BLOCK bytes per Q1 block per rowblock.
        packed_bytes_per_rb = q1a8_layout.packed_bytes_per_rowblock(k)
        weight_nbytes = q1a8_layout.packed_nbytes(rows, k)
        act_nbytes = cols * k * F32_BYTES
        dst_nbytes = rows * cols * F32_BYTES

        # Acts are small (cols * k floats); resolve once. Weights are
        # resolved per-chunk inside the loop below, because the whole-tensor
        # scratch copy would not fit in CMA for big matmuls (lm_head's
        # token_embd.weight is ~42 MiB; CMA has ~15 MiB free after the model
        # is loaded). Per-chunk weights are at most rows_per_chunk *
        # packed_bytes_per_rb ≈ a few MiB which always fits.
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
        rowblock_stream_nbytes = q1a8_layout.STREAM_PER_Q1_BLOCK * blocks_per_row
        rowblocks_per_col = (rows + rows_per_block - 1) // rows_per_block

        # Cap the stream/result buffers to ~4 MiB so huge matmuls (lm_head:
        # rows=151669 → 88 MiB stream) split into manageable chunks of
        # ``rows_per_chunk`` rows. Layer matmuls (rows ≤ 6144) fit in one
        # chunk and pay no extra kernel-invocation overhead.
        max_stream_nbytes = 4 * 1024 * 1024
        max_rowblocks_per_chunk = min(
            rowblocks_per_col,
            max(1, max_stream_nbytes // rowblock_stream_nbytes),
        )
        rows_per_chunk = max_rowblocks_per_chunk * rows_per_block
        chunk_stream_nbytes = max_rowblocks_per_chunk * rowblock_stream_nbytes
        chunk_result_nbytes = max_rowblocks_per_chunk * rows_per_block * F32_BYTES

        import numpy as np

        self._np = np
        self._ensure_buffers(chunk_stream_nbytes, chunk_result_nbytes)

        act_quants = np.empty(k, dtype=np.int8)
        act_scale_bits = np.empty(k // Q8_BLOCK, dtype=np.uint16)

        quants_ptr = act_quants.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        scales_ptr = act_scale_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        stream_ptr = self._stream_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        out = bytearray(dst_nbytes)
        total_cycles = 0

        for col in range(cols):
            col_acts_offset = col * k * F32_BYTES

            with timer.section("quantize"):
                acts_ptr = ctypes.cast(
                    acts_addr + col_acts_offset,
                    ctypes.POINTER(ctypes.c_float),
                )
                rc = self._c_quantize(acts_ptr, ctypes.c_uint32(k),
                                      quants_ptr, scales_ptr)
                if rc != 0:
                    raise RuntimeError(f"bonsai_quantize_q8_0_pl rc={rc}")

            row_chunk_start = 0
            rowblock_chunk_start = 0
            while row_chunk_start < rows:
                chunk_rows = min(rows_per_chunk, rows - row_chunk_start)
                chunk_rowblocks = (chunk_rows + rows_per_block - 1) // rows_per_block
                chunk_stream = chunk_rowblocks * rowblock_stream_nbytes
                chunk_result = chunk_rowblocks * rows_per_block * F32_BYTES

                # Packed weights are rowblock-major; offset by full rowblocks
                # processed so far (chunking always aligns to a rowblock).
                chunk_weight_offset = (
                    weights_base_offset
                    + rowblock_chunk_start * packed_bytes_per_rb
                )
                chunk_weight_nbytes = chunk_rowblocks * packed_bytes_per_rb
                chunk_weights_addr = self._resolve_slab_pointer(
                    allocator,
                    weights_handle,
                    chunk_weight_offset,
                    chunk_weight_nbytes,
                    "weights",
                )
                chunk_weights_ptr = ctypes.cast(
                    chunk_weights_addr,
                    ctypes.POINTER(ctypes.c_uint8),
                )

                with timer.section("merge"):
                    # rows passed to merge_acts is the logical chunk_rows;
                    # internally it rounds up to whole rowblocks and zero-
                    # pads inactive lanes via the same pre-packed zeros.
                    rc = self._c_merge(
                        chunk_weights_ptr,
                        quants_ptr,
                        scales_ptr,
                        ctypes.c_uint32(chunk_rows),
                        ctypes.c_uint32(k),
                        stream_ptr,
                    )
                    if rc != 0:
                        raise RuntimeError(f"bonsai_q1a8_merge_acts rc={rc}")

                with timer.section("flush"):
                    self._stream_buf.flush()

                with timer.section("compute"):
                    cycles = self._run_matmul(
                        chunk_stream,
                        chunk_result,
                        blocks_per_row,
                        chunk_rowblocks,
                        timer,
                    )
                total_cycles += cycles

                with timer.section("result_copy"):
                    # First `chunk_rows` fp32 are meaningful; the rest are
                    # zero-padded inactive lanes inside the last rowblock.
                    result_view = self._np.frombuffer(
                        self._result_buf, dtype=self._np.uint8,
                        count=chunk_rows * F32_BYTES,
                    )
                    out_base = (col * rows + row_chunk_start) * F32_BYTES
                    out[out_base : out_base + chunk_rows * F32_BYTES] = result_view.tobytes()

                row_chunk_start += chunk_rows
                rowblock_chunk_start += chunk_rowblocks

        with timer.section("write"):
            allocator.write(
                _required(op, F_DST),
                _optional_int(op, F_DST_OFFSET),
                out,
            )
        timer.add("bytes_written", dst_nbytes)
        timer.add("matmul_cols", cols)
        timer.add("rowblocks", cols * rowblocks_per_col)
        timer.add("dma_bytes_read", cols * rowblocks_per_col * rowblock_stream_nbytes)
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
            raise RuntimeError(f"q1a8 rowblock kernel version mismatch: got {got_version}")

    def _read_rows_per_block(self) -> int:
        rows_per_block = int(self._kernel.read(REG_ROWS))
        if rows_per_block != ROWS_PER_BLOCK:
            raise RuntimeError(f"q1a8 rowblock lanes mismatch: got {rows_per_block}")
        self._rows_per_block = rows_per_block
        return rows_per_block

    def _ensure_buffers(self, stream_nbytes: int, result_nbytes: int) -> None:
        from pynq import allocate

        if self._stream_buf is None or self._stream_buf_size < stream_nbytes:
            if self._stream_buf is not None:
                self._stream_buf.freebuffer()
            self._stream_buf = allocate(shape=(stream_nbytes,), dtype=self._np.uint8)
            self._stream_buf_size = stream_nbytes

        if self._result_buf is None or self._result_buf_size < result_nbytes:
            if self._result_buf is not None:
                self._result_buf.freebuffer()
            self._result_buf = allocate(shape=(result_nbytes,), dtype=self._np.uint8)
            self._result_buf_size = result_nbytes

    def _run_matmul(
        self,
        stream_nbytes: int,
        result_nbytes: int,
        num_q1_blocks: int,
        num_rowblocks: int,
        timer: Timer,
    ) -> int:
        """Drive one column's matmul: configure kernel, start both DMA channels,
        wait for both to drain. Returns cycle count of the run."""
        assert self._stream_buf is not None and self._result_buf is not None

        stream_view = self._stream_buf[:stream_nbytes]
        result_view = self._result_buf[:result_nbytes]

        with timer.section("kernel_setup"):
            self._kernel.write(REG_NUM_Q1_BLOCKS, num_q1_blocks)
            self._kernel.write(REG_NUM_ROWBLOCKS, num_rowblocks)

        with timer.section("recv_start"):
            # Arm S2MM first so the kernel cannot deadlock when it starts
            # emitting after the first rowblock.
            self._dma.recvchannel.transfer(result_view)

        with timer.section("kernel_start"):
            self._kernel.write(REG_CTRL, CTRL_START)

        with timer.section("send_start"):
            self._dma.sendchannel.transfer(stream_view)

        with timer.section("send_wait"):
            self._dma.sendchannel.wait()

        with timer.section("recv_wait"):
            self._dma.recvchannel.wait()

        with timer.section("poll"):
            status = 0
            for _ in range(POLL_LIMIT):
                status = self._kernel.read(REG_STATUS)
                if status & STATUS_DONE:
                    break
            else:
                raise RuntimeError(
                    f"q1a8 kernel never reported done (status=0x{status:08x})")

        with timer.section("result_invalidate"):
            result_view.invalidate()

        return int(self._kernel.read(REG_CYCLES))
