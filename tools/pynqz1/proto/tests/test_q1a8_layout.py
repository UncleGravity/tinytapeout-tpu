"""Verify the canonical Q1A8 layout matches the existing on-the-wire format.

Two checks:

1. proto/q1a8_layout.h and proto/q1a8_layout.py agree on every shared
   constant (mirrors the proto/ops parity test).

2. The Python reference pack_weights + merge_acts produces byte-identical
   output to the existing board/kernels/ps/native.c pack function (which
   the current bitstream already accepts). This makes the new layout a
   no-op for the FPGA: same bytes on the wire, just produced differently.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from proto import q1a8_layout as L

HEADER = Path(__file__).resolve().parents[1] / "q1a8_layout.h"


_DEFINE = re.compile(r"#define\s+(BONSAI_\w+)\s+([^\n]+)")
_C_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def _parse_macros() -> dict[str, int]:
    """Resolve #define macros from q1a8_layout.h to ints.

    Simple recursive substitution — the header only uses arithmetic on
    previously-defined macros, so a few passes converge.
    """
    text = _C_COMMENT.sub("", HEADER.read_text())
    # collapse line continuations: "\\\n" -> " "
    text = re.sub(r"\\\n", " ", text)
    raw: dict[str, str] = {}
    for match in _DEFINE.finditer(text):
        raw[match.group(1)] = match.group(2).strip()

    resolved: dict[str, int] = {}
    progress = True
    while progress:
        progress = False
        for name, expr in raw.items():
            if name in resolved:
                continue
            expanded = expr
            for known, value in resolved.items():
                expanded = re.sub(rf"\b{known}\b", str(value), expanded)
            try:
                resolved[name] = int(eval(expanded, {"__builtins__": {}}, {}))
                progress = True
            except Exception:
                continue
    return resolved


def test_header_and_python_constants_agree():
    header = _parse_macros()
    expected = {
        "BONSAI_Q1_BLOCK": 128,
        "BONSAI_Q1_BLOCK_BYTES": 18,
        "BONSAI_Q8_BLOCK": 32,
        "BONSAI_Q1A8_ROWS_PER_BLOCK": L.ROWS_PER_BLOCK,
        "BONSAI_Q1A8_SCALE_BEATS": L.SCALE_BEATS,
        "BONSAI_Q1A8_WBITS_BEATS": L.WBITS_BEATS,
        "BONSAI_Q1A8_Q8_SUBBLOCKS": L.Q8_SUBBLOCKS,
        "BONSAI_Q1A8_SCALES_BYTES": L.SCALES_BYTES,
        "BONSAI_Q1A8_ACTS_BYTES": L.ACTS_BYTES,
        "BONSAI_Q1A8_ACT_SCALE_BYTES": L.ACT_SCALE_BYTES,
        "BONSAI_Q1A8_WBITS_BYTES": L.WBITS_BYTES,
        "BONSAI_Q1A8_SUBBLOCK_STREAM_BYTES": L.SUBBLOCK_STREAM_BYTES,
        "BONSAI_Q1A8_STREAM_PER_Q1_BLOCK": L.STREAM_PER_Q1_BLOCK,
        "BONSAI_Q1A8_PACKED_PER_Q1_BLOCK": L.PACKED_PER_Q1_BLOCK,
    }
    missing = sorted(set(expected) - set(header))
    assert not missing, f"in q1a8_layout.py but not q1a8_layout.h: {missing}"
    mismatched = {
        name: (expected[name], header[name])
        for name in expected
        if header[name] != expected[name]
    }
    assert not mismatched, f"value mismatch: {mismatched}"


def test_packed_size_matches_q1_0_source_size():
    """Same total bytes on the board — the whole reason this design works."""
    rows, k = 32, 256
    assert L.packed_nbytes(rows, k) == rows * (k // 128) * 18


def test_stream_size_matches_existing_packer():
    """The current ps/native.c pack function and the new layout produce
    streams of the same total size for the same matmul dims."""
    rows, k, cols = 16, 256, 1
    # 2 rowblocks × 2 q1_blocks × 304 bytes
    assert L.stream_nbytes(rows, k, cols) == 2 * 2 * 304


def test_pack_then_merge_matches_legacy_packer():
    """The new pack(Q1_0) + merge(packed, acts) must produce identical
    bytes to the legacy bonsai_pack_matmul_q1a8_stream. This is the
    bit-level contract that lets the existing bitstream keep working."""
    from board.kernels.pl.matmul_q1a8 import (
        _pack_rowblock_into,
        _quantize_q8_0,
        _rowblock_nbytes,
    )

    rows, k = 24, 256  # 3 rowblocks, partial last (rows=24, ROWS_PER_BLOCK=8 → 3 full)
    # actually 24 / 8 = exactly 3, no partial. Let's force a partial:
    rows = 21

    rng_state = 0x12345
    def lcg() -> int:
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0xFFFFFFFF
        return rng_state

    bpr = k // 128
    weight_row_bytes = bpr * 18
    q1_0 = bytearray(rows * weight_row_bytes)
    for i in range(len(q1_0)):
        q1_0[i] = lcg() & 0xFF

    # Build per-column acts.
    acts = tuple((lcg() & 0xFF) / 32.0 - 4.0 for _ in range(k))
    quants, scales = _quantize_q8_0(acts)
    scale_bytes = b"".join(struct.pack("<H", s) for s in scales)

    # New path: pack then merge.
    packed = L.pack_weights(bytes(q1_0), rows, k)
    new_stream = L.merge_acts(packed, bytes(q & 0xFF for q in quants), scale_bytes, rows, k)

    # Legacy path: pack rowblock by rowblock.
    rowblocks = (rows + 7) // 8
    rb_nbytes = _rowblock_nbytes(k)
    legacy = bytearray(rowblocks * rb_nbytes)
    for rb in range(rowblocks):
        row_start = rb * 8
        row_count = min(8, rows - row_start)
        out = bytearray(rb_nbytes)
        _pack_rowblock_into(
            out, bytes(q1_0), row_start, row_count, weight_row_bytes,
            list(quants), list(scales), k,
        )
        legacy[rb * rb_nbytes : (rb + 1) * rb_nbytes] = out

    assert bytes(new_stream) == bytes(legacy), (
        f"new stream differs from legacy at first diff index "
        f"{next((i for i in range(len(new_stream)) if new_stream[i] != legacy[i]), 'none')}"
    )
