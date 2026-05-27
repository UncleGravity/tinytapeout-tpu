"""PL-backed MATMUL_Q1A8 driver for the rowblock bitstream."""

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
ROWS_PER_BLOCK = 8

REG_ID = 0x00
REG_VERSION = 0x04
REG_CTRL = 0x08
REG_STATUS = 0x0C
REG_NUM_Q1_BLOCKS = 0x10
REG_ROW_COUNT = 0x14
REG_RESULT_INDEX = 0x18
REG_RESULT = 0x1C
REG_CYCLES = 0x20
REG_ROWS = 0x24

CTRL_START = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID = 0xB05A_2000
EXPECTED_VERSION = 2
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
    """Match the PS kernel's Q8_0 activation quantization."""
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
    """Pack one rowblock into the RTL rowblock stream format."""
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


class PLMatmulQ1A8:
    name = GOP_MATMUL_Q1A8
    backend = "pl"

    def __init__(self, overlay):
        self._dma = overlay.axi_dma_0
        self._kernel = overlay.q1a8_kernel_top_0
        self._buf = None
        self._buf_size = 0
        self._np = None
        self._rows_per_block = ROWS_PER_BLOCK

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
        rows_per_block = self._read_rows_per_block()
        packed_nbytes = _rowblock_nbytes(k, rows_per_block)
        self._ensure_buffer(packed_nbytes)

        packed = bytearray(packed_nbytes)
        out = bytearray(dst_nbytes)
        total_cycles = 0
        rowblocks = 0

        for col in range(cols):
            with timer.section("quantize"):
                act_values = struct.unpack_from(f"<{k}f", acts, col * k * F32_BYTES)
                act_quants, act_scale_bits = _quantize_q8_0(act_values)

            for row_start in range(0, rows, rows_per_block):
                row_count = min(rows_per_block, rows - row_start)
                with timer.section("pack"):
                    _pack_rowblock_into(
                        packed,
                        weights,
                        row_start,
                        row_count,
                        weight_row_bytes,
                        act_quants,
                        act_scale_bits,
                        k,
                        rows_per_block,
                    )

                result_bits, cycles = self._run_rowblock(packed, row_count, blocks_per_row, timer)
                total_cycles += cycles
                rowblocks += 1
                for lane, bits in enumerate(result_bits):
                    struct.pack_into(
                        "<I",
                        out,
                        (col * rows + row_start + lane) * F32_BYTES,
                        bits,
                    )

        with timer.section("write"):
            allocator.write(
                _required(op, F_DST),
                _optional_int(op, F_DST_OFFSET),
                out,
            )
        timer.add("bytes_written", dst_nbytes)
        timer.add("rowblocks", rowblocks)
        timer.add("dma_bytes_read", rowblocks * packed_nbytes)
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

    def _run_rowblock(
        self,
        packed: bytearray,
        row_count: int,
        num_q1_blocks: int,
        timer: Timer,
    ) -> tuple[list[int], int]:
        nbytes = len(packed)
        assert self._buf is not None
        assert self._np is not None

        with timer.section("dma_load"):
            self._buf[:nbytes] = self._np.frombuffer(packed, dtype=self._np.uint8)
            self._buf.flush()

        with timer.section("kernel_setup"):
            self._kernel.write(REG_NUM_Q1_BLOCKS, num_q1_blocks)
            self._kernel.write(REG_ROW_COUNT, row_count)

        view = self._buf[:nbytes]
        with timer.section("dma_start"):
            self._dma.sendchannel.transfer(view)

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
                raise RuntimeError(f"q1a8 rowblock kernel never reported done (status=0x{status:08x})")

        results: list[int] = []
        with timer.section("result_read"):
            for lane in range(row_count):
                self._kernel.write(REG_RESULT_INDEX, lane)
                results.append(int(self._kernel.read(REG_RESULT)) & 0xFFFF_FFFF)
            cycles = int(self._kernel.read(REG_CYCLES))
        return results, cycles
