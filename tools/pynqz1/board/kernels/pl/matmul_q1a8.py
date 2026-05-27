"""PL-backed MATMUL_Q1A8 driver for the current single-cell bitstream.

This is intentionally a bring-up driver, not the final high-throughput
matmul architecture. It packs one output cell at a time into the 48-byte
sub-block stream consumed by ``q1a8_kernel_top`` and writes each fp32 result
back into the daemon-owned destination tensor.
"""

from __future__ import annotations

import math
import struct
from typing import Any

from board.memory.allocator import AllocatorError, TensorAllocator
from board.profiling.timer import Timer
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
SUBBLOCK_BYTES = 48

REG_ID = 0x00
REG_VERSION = 0x04
REG_CTRL = 0x08
REG_STATUS = 0x0C
REG_NUM_SUBBLOCKS = 0x10
REG_RESULT = 0x14
REG_CYCLES = 0x18

CTRL_START = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID = 0xB05A_1000
EXPECTED_VERSION = 1
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


def _lround_like_native(value: float) -> int:
    if value >= 0.0:
        return int(value + 0.5)
    return int(value - 0.5)


def _fp16_float_to_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def _quantize_q8_0(values: tuple[float, ...]) -> tuple[list[int], list[int]]:
    """Match the PS kernel's Q8_0 activation quantization.

    The quantized int8 values use the unrounded scale, while the scale sent to
    hardware is the fp16 representation, exactly like ``native.c``.
    """
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


def _pack_cell_into(
    out: bytearray,
    weight_row: bytes,
    act_quants: list[int],
    act_scale_bits: list[int],
    k: int,
) -> None:
    """Pack one output cell into the RTL's 48-byte-per-Q8-block stream."""
    expected_len = (k // Q8_BLOCK) * SUBBLOCK_BYTES
    if len(out) != expected_len:
        raise ValueError(f"packed buffer must be {expected_len} bytes")
    if len(weight_row) != (k // Q1_BLOCK) * Q1_BLOCK_BYTES:
        raise ValueError("weight row has wrong length")

    cursor = 0
    blocks_per_row = k // Q1_BLOCK
    for q1_index in range(blocks_per_row):
        block_offset = q1_index * Q1_BLOCK_BYTES
        weight_scale_bits = struct.unpack_from("<H", weight_row, block_offset)[0]
        bits_offset = block_offset + 2
        q1_base = q1_index * Q1_BLOCK

        for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
            q8_base = q1_base + q8_local
            weight_bits = int.from_bytes(
                weight_row[bits_offset + q8_local // 8 : bits_offset + q8_local // 8 + 4],
                "little",
            )
            struct.pack_into("<II", out, cursor, weight_bits, 0)
            cursor += 8

            out[cursor : cursor + Q8_BLOCK] = bytes(
                quant & 0xFF for quant in act_quants[q8_base : q8_base + Q8_BLOCK]
            )
            cursor += Q8_BLOCK

            struct.pack_into(
                "<HHI",
                out,
                cursor,
                weight_scale_bits,
                act_scale_bits[q8_base // Q8_BLOCK],
                0,
            )
            cursor += 8


class PLMatmulQ1A8:
    name = GOP_MATMUL_Q1A8
    backend = "pl"

    def __init__(self, overlay):
        self._dma = overlay.axi_dma_0
        self._kernel = overlay.q1a8_kernel_top_0
        self._buf = None
        self._buf_size = 0
        self._np = None

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

        blocks_per_row = k // Q1_BLOCK
        weight_row_bytes = blocks_per_row * Q1_BLOCK_BYTES
        weight_nbytes = rows * weight_row_bytes
        act_nbytes = cols * k * F32_BYTES
        dst_nbytes = rows * cols * F32_BYTES
        packed_nbytes = (k // Q8_BLOCK) * SUBBLOCK_BYTES

        with timer.section("read"):
            weights = allocator.read(
                _required(op, F_WEIGHTS),
                _optional_int(op, F_WEIGHTS_OFFSET),
                weight_nbytes,
            )
            acts = allocator.read(
                _required(op, F_ACTS),
                _optional_int(op, F_ACTS_OFFSET),
                act_nbytes,
            )
        timer.add("bytes_read", weight_nbytes + act_nbytes)

        self._check_kernel_id()
        self._ensure_buffer(packed_nbytes)
        packed = bytearray(packed_nbytes)
        out = bytearray(dst_nbytes)
        total_cycles = 0
        cells = rows * cols

        for col in range(cols):
            with timer.section("quantize"):
                act_values = struct.unpack_from(f"<{k}f", acts, col * k * F32_BYTES)
                act_quants, act_scale_bits = _quantize_q8_0(act_values)

            for row in range(rows):
                row_start = row * weight_row_bytes
                with timer.section("pack"):
                    _pack_cell_into(
                        packed,
                        weights[row_start : row_start + weight_row_bytes],
                        act_quants,
                        act_scale_bits,
                        k,
                    )

                result_bits, cycles = self._run_cell(packed, timer)
                total_cycles += cycles
                struct.pack_into("<I", out, (col * rows + row) * F32_BYTES, result_bits)

        with timer.section("write"):
            allocator.write(
                _required(op, F_DST),
                _optional_int(op, F_DST_OFFSET),
                out,
            )
        timer.add("bytes_written", dst_nbytes)
        timer.add("cells", cells)
        timer.add("dma_bytes_read", cells * packed_nbytes)
        timer.add("kernel_cycles", total_cycles)

    def _check_kernel_id(self) -> None:
        got_id = self._kernel.read(REG_ID)
        got_version = self._kernel.read(REG_VERSION)
        if isinstance(got_id, int) and got_id != EXPECTED_ID:
            raise RuntimeError(f"q1a8 kernel ID mismatch: got 0x{got_id:08x}")
        if isinstance(got_version, int) and got_version != EXPECTED_VERSION:
            raise RuntimeError(f"q1a8 kernel version mismatch: got {got_version}")

    def _ensure_buffer(self, nbytes: int) -> None:
        if self._buf is not None and self._buf_size >= nbytes:
            return
        if self._buf is not None:
            self._buf.freebuffer()

        import numpy as np
        from pynq import allocate

        self._np = np
        self._buf = allocate(shape=(nbytes,), dtype=np.uint8)
        self._buf_size = nbytes

    def _run_cell(self, packed: bytearray, timer: Timer) -> tuple[int, int]:
        nbytes = len(packed)
        assert self._buf is not None
        assert self._np is not None

        with timer.section("dma_load"):
            self._buf[:nbytes] = self._np.frombuffer(packed, dtype=self._np.uint8)
            self._buf.flush()

        with timer.section("kernel_setup"):
            self._kernel.write(REG_NUM_SUBBLOCKS, nbytes // SUBBLOCK_BYTES)

        view = self._buf[:nbytes]
        with timer.section("dma_start"):
            self._dma.sendchannel.transfer(view)

        # Arm DMA first, then start the kernel. The DMA may deliver the first
        # packed sub-block and stall until the streamer becomes ready; this
        # keeps PS setup time out of the kernel CYCLES register.
        with timer.section("kernel_start"):
            self._kernel.write(REG_CTRL, CTRL_START)

        with timer.section("dma_wait"):
            self._dma.sendchannel.wait()

        with timer.section("poll"):
            status = 0
            for _ in range(POLL_LIMIT):
                status = self._kernel.read(REG_STATUS)
                if status & STATUS_DONE:
                    break
            else:
                raise RuntimeError(f"q1a8 kernel never reported done (status=0x{status:08x})")

        with timer.section("result_read"):
            result_bits = int(self._kernel.read(REG_RESULT)) & 0xFFFF_FFFF
            cycles = int(self._kernel.read(REG_CYCLES))
        return result_bits, cycles
