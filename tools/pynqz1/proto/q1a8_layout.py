"""Canonical Q1A8 wire/packed layout.

Mirror of ``proto/q1a8_layout.h``. Keep the constants and the layout doc-
comment in sync — the parity test in ``proto/tests/test_q1a8_layout.py``
verifies the byte counts match the header.

The packed layout (what the host stores on the board) carries only the
weight portion of the AXIS stream and is exactly the same total size as
the Q1_0 source: 8 × Q1_BLOCK_BYTES bytes per Q1 block per rowblock. The
runtime merge function then walks packed[rb][q1] + acts[col][q8] in
lock-step to produce the full 304-byte stream chunk the PL kernel reads.
"""

from __future__ import annotations

import struct

from proto.ops import Q1_BLOCK, Q1_BLOCK_BYTES, Q8_BLOCK

# -- layout constants ------------------------------------------------------

ROWS_PER_BLOCK = 8
SCALE_BEATS = (ROWS_PER_BLOCK + 3) // 4              # 2
WBITS_BEATS = (ROWS_PER_BLOCK + 1) // 2              # 4
Q8_SUBBLOCKS = Q1_BLOCK // Q8_BLOCK                  # 4

SCALES_BYTES = SCALE_BEATS * 8                       # 16
ACTS_BYTES = Q8_BLOCK                                # 32
ACT_SCALE_BYTES = 8
WBITS_BYTES = WBITS_BEATS * 8                        # 32
SUBBLOCK_STREAM_BYTES = ACTS_BYTES + ACT_SCALE_BYTES + WBITS_BYTES  # 72
STREAM_PER_Q1_BLOCK = SCALES_BYTES + Q8_SUBBLOCKS * SUBBLOCK_STREAM_BYTES  # 304
PACKED_PER_Q1_BLOCK = SCALES_BYTES + Q8_SUBBLOCKS * WBITS_BYTES            # 144
ACTS_PER_Q1_BLOCK = Q8_SUBBLOCKS * (ACTS_BYTES + ACT_SCALE_BYTES)          # 160

assert PACKED_PER_Q1_BLOCK == ROWS_PER_BLOCK * Q1_BLOCK_BYTES, (
    "packed weight layout size must equal Q1_0 source size; layout changed?"
)


# -- size helpers ----------------------------------------------------------


def rowblocks_for(rows: int) -> int:
    return (rows + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK


def blocks_per_row(k: int) -> int:
    if k % Q1_BLOCK != 0:
        raise ValueError(f"k={k} must be a multiple of Q1_BLOCK={Q1_BLOCK}")
    return k // Q1_BLOCK


def packed_nbytes(rows: int, k: int) -> int:
    return rowblocks_for(rows) * blocks_per_row(k) * PACKED_PER_Q1_BLOCK


def stream_nbytes(rows: int, k: int, cols: int = 1) -> int:
    return cols * rowblocks_for(rows) * blocks_per_row(k) * STREAM_PER_Q1_BLOCK


def stream_bytes_per_rowblock(k: int) -> int:
    return blocks_per_row(k) * STREAM_PER_Q1_BLOCK


def packed_bytes_per_rowblock(k: int) -> int:
    return blocks_per_row(k) * PACKED_PER_Q1_BLOCK


def acts_stream_nbytes(k: int) -> int:
    """v4 acts stream size per matmul column."""
    return blocks_per_row(k) * ACTS_PER_Q1_BLOCK


# -- reference packer + merger --------------------------------------------


def pack_weights(q1_0_weights: bytes, rows: int, k: int) -> bytes:
    """Repack Q1_0 row-major weights into the packed AXIS layout.

    Q1_0 source: rows × (k / Q1_BLOCK) × Q1_BLOCK_BYTES, row-major.
    Each Q1_BLOCK_BYTES = 2 bytes fp16 scale + 16 bytes wbits.
    """
    bpr = blocks_per_row(k)
    weight_row_bytes = bpr * Q1_BLOCK_BYTES
    if len(q1_0_weights) < rows * weight_row_bytes:
        raise ValueError("q1_0_weights too short for the requested rows/k")

    rbs = rowblocks_for(rows)
    out = bytearray(rbs * bpr * PACKED_PER_Q1_BLOCK)
    cursor = 0

    for rb in range(rbs):
        row_start = rb * ROWS_PER_BLOCK
        row_count = min(ROWS_PER_BLOCK, rows - row_start)
        for q1 in range(bpr):
            # weight_scales: SCALE_BEATS beats of 4 packed fp16 scales each.
            for beat in range(SCALE_BEATS):
                word = 0
                for local in range(4):
                    lane = beat * 4 + local
                    scale_bits = 0
                    if lane < row_count:
                        off = (row_start + lane) * weight_row_bytes + q1 * Q1_BLOCK_BYTES
                        scale_bits = q1_0_weights[off] | (q1_0_weights[off + 1] << 8)
                    word |= scale_bits << (local * 16)
                struct.pack_into("<Q", out, cursor, word)
                cursor += 8
            # wbits, 4 sub-blocks of WBITS_BEATS each (2 u32 per beat).
            for sub in range(Q8_SUBBLOCKS):
                for beat in range(WBITS_BEATS):
                    word = 0
                    for local in range(2):
                        lane = beat * 2 + local
                        bits = 0
                        if lane < row_count:
                            base = (
                                (row_start + lane) * weight_row_bytes
                                + q1 * Q1_BLOCK_BYTES
                                + 2  # skip fp16 scale
                                + sub * (Q8_BLOCK // 8)
                            )
                            bits = (
                                q1_0_weights[base]
                                | (q1_0_weights[base + 1] << 8)
                                | (q1_0_weights[base + 2] << 16)
                                | (q1_0_weights[base + 3] << 24)
                            )
                        word |= bits << (local * 32)
                    struct.pack_into("<Q", out, cursor, word)
                    cursor += 8

    return bytes(out)


def pack_acts(
    act_quants: bytes,
    act_scale_bits: bytes,
    k: int,
) -> bytes:
    """Pack one column's acts + fp16 scales into the v4 acts wire stream.

    act_quants:     k bytes of int8 (one column post-Q8_0)
    act_scale_bits: (k / Q8_BLOCK) × 2 bytes, fp16 LE
    """
    if len(act_quants) != k:
        raise ValueError("act_quants must be k bytes")
    bpr = blocks_per_row(k)
    if len(act_scale_bits) != bpr * Q8_SUBBLOCKS * 2:
        raise ValueError("act_scale_bits length mismatch")

    out = bytearray(acts_stream_nbytes(k))
    s = 0
    for q1 in range(bpr):
        for sub in range(Q8_SUBBLOCKS):
            q8_idx = q1 * Q8_SUBBLOCKS + sub
            a_off = q8_idx * Q8_BLOCK
            out[s : s + ACTS_BYTES] = act_quants[a_off : a_off + ACTS_BYTES]
            s += ACTS_BYTES
            scale_lo = act_scale_bits[q8_idx * 2]
            scale_hi = act_scale_bits[q8_idx * 2 + 1]
            struct.pack_into("<Q", out, s, scale_lo | (scale_hi << 8))
            s += ACT_SCALE_BYTES
    return bytes(out)


def merge_acts(
    packed_weights: bytes,
    act_quants: bytes,
    act_scale_bits: bytes,
    rows: int,
    k: int,
) -> bytes:
    """Produce one column's AXIS stream from packed weights + Q8 acts.

    act_quants: k signed int8 (one column post-Q8_0)
    act_scale_bits: (k / Q8_BLOCK) × 2 bytes, fp16 LE
    """
    bpr = blocks_per_row(k)
    rbs = rowblocks_for(rows)
    if len(packed_weights) != rbs * bpr * PACKED_PER_Q1_BLOCK:
        raise ValueError("packed_weights size mismatch")
    if len(act_quants) != k:
        raise ValueError("act_quants must be k bytes")
    if len(act_scale_bits) != bpr * Q8_SUBBLOCKS * 2:
        raise ValueError("act_scale_bits length mismatch")

    out = bytearray(rbs * bpr * STREAM_PER_Q1_BLOCK)
    p = 0  # packed cursor
    s = 0  # stream cursor

    for _rb in range(rbs):
        for q1 in range(bpr):
            # copy weight_scales
            out[s : s + SCALES_BYTES] = packed_weights[p : p + SCALES_BYTES]
            s += SCALES_BYTES
            p += SCALES_BYTES
            for sub in range(Q8_SUBBLOCKS):
                # acts (broadcast across rows; identical each rowblock)
                a_off = q1 * Q1_BLOCK + sub * Q8_BLOCK
                out[s : s + ACTS_BYTES] = act_quants[a_off : a_off + ACTS_BYTES]
                s += ACTS_BYTES
                # act_scale beat (low 16 bits used, rest zero)
                scale_index = (q1 * Q1_BLOCK + sub * Q8_BLOCK) // Q8_BLOCK
                scale_lo = act_scale_bits[scale_index * 2]
                scale_hi = act_scale_bits[scale_index * 2 + 1]
                struct.pack_into("<Q", out, s, scale_lo | (scale_hi << 8))
                s += ACT_SCALE_BYTES
                # wbits
                out[s : s + WBITS_BYTES] = packed_weights[p : p + WBITS_BYTES]
                s += WBITS_BYTES
                p += WBITS_BYTES

    return bytes(out)
