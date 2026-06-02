"""PL MATMUL_Q1A8 driver tests that do not require a PYNQ board."""

from __future__ import annotations

import ctypes
import random
import struct
from types import SimpleNamespace

from board.kernels import pl
from board.kernels.pl import matmul_q1a8
from board.kernels.ps.native import load_lib
from board.kernels.registry import KernelRegistry
from proto.ops import GOP_COPY, GOP_MATMUL_Q1A8, Q1_BLOCK, Q1_BLOCK_BYTES, Q8_BLOCK
from tests.golden import kernels as golden


def test_quantize_q8_0_matches_golden():
    values = tuple((i - 17) / 13.0 for i in range(Q8_BLOCK * 2))
    quants, scale_bits = matmul_q1a8._quantize_q8_0(values)
    exp_quants, exp_scales = golden.quantize_q8_0(values)

    assert quants == exp_quants
    assert scale_bits == [
        struct.unpack("<H", struct.pack("<e", scale))[0]
        for scale in exp_scales
    ]


def test_pack_rowblock_matches_axis_format():
    row_count = 3
    weight_row_bytes = Q1_BLOCK_BYTES
    weights = bytearray()
    row_bits = []
    for row in range(row_count):
        weight_scale = 0x1234 + row
        bits = bytes((row * 17 + i) & 0xFF for i in range(Q1_BLOCK // 8))
        row_bits.append(bits)
        weights += struct.pack("<H", weight_scale) + bits

    act_quants = [i - 64 for i in range(Q1_BLOCK)]
    act_scale_bits = [0x2000, 0x2001, 0x2002, 0x2003]
    packed = bytearray(matmul_q1a8._rowblock_nbytes(Q1_BLOCK))

    matmul_q1a8._pack_rowblock_into(
        packed,
        bytes(weights),
        0,
        row_count,
        weight_row_bytes,
        act_quants,
        act_scale_bits,
        Q1_BLOCK,
    )

    scale_beats = (matmul_q1a8.ROWS_PER_BLOCK + 3) // 4
    wbits_beats = (matmul_q1a8.ROWS_PER_BLOCK + 1) // 2
    assert packed[:8] == struct.pack("<Q", 0x1234 | (0x1235 << 16) | (0x1236 << 32))
    assert packed[8 : scale_beats * 8] == bytes((scale_beats - 1) * 8)

    cursor = scale_beats * 8
    for q8_index in range(Q1_BLOCK // Q8_BLOCK):
        bit_base = q8_index * (Q8_BLOCK // 8)
        assert packed[cursor : cursor + Q8_BLOCK] == bytes(
            value & 0xFF for value in act_quants[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
        )
        cursor += Q8_BLOCK

        assert packed[cursor : cursor + 8] == struct.pack("<Q", act_scale_bits[q8_index])
        cursor += 8

        for beat in range(wbits_beats):
            word = 0
            for local in range(2):
                lane = beat * 2 + local
                bits = 0
                if lane < row_count:
                    bits = int.from_bytes(row_bits[lane][bit_base : bit_base + 4], "little")
                word |= bits << (local * 32)
            assert packed[cursor : cursor + 8] == struct.pack("<Q", word)
            cursor += 8

    assert cursor == len(packed)


def test_register_all_registers_matmul_only_for_matmul_overlay():
    registry = KernelRegistry()
    # v4 bitstream wires two DMAs: axi_dma_0 (weights + results) and
    # axi_dma_1 (acts MM2S only).
    overlay = SimpleNamespace(
        ip_dict={
            "axi_dma_0": {},
            "axi_dma_1": {},
            "q1a8_kernel_top_0": {},
        },
        axi_dma_0=SimpleNamespace(),
        axi_dma_1=SimpleNamespace(),
        q1a8_kernel_top_0=SimpleNamespace(),
    )

    pl.register_all(registry, overlay)

    assert isinstance(registry.get(GOP_MATMUL_Q1A8), matmul_q1a8.PLMatmulQ1A8)
    assert GOP_COPY not in registry


def test_extent_remaining_respects_slab_boundary(allocator):
    """_extent_remaining reports bytes to the end of the current slab extent,
    so weight chunks can be sized to stay single-extent (fix #2)."""
    nbytes = 600 * 1024  # > slab_bytes (256 KiB) → spans multiple extents
    rec = allocator.allocate(nbytes, shape=[nbytes], dtype="u8")
    exts = allocator.extents(rec.handle)
    assert len(exts) > 1

    remaining = matmul_q1a8.PLMatmulQ1A8._extent_remaining
    first = exts[0].nbytes
    assert remaining(allocator, rec.handle, 0) == first
    assert remaining(allocator, rec.handle, first - 10) == 10
    assert remaining(allocator, rec.handle, first) == exts[1].nbytes


# -- C entry point tests vs Python reference -----------------------------


def _bind_c_quantize(lib: ctypes.CDLL):
    fn = lib.bonsai_quantize_q8_0_pl
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    fn.restype = ctypes.c_int
    return fn


def _bind_c_pack(lib: ctypes.CDLL):
    fn = lib.bonsai_pack_matmul_q1a8_stream
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    fn.restype = ctypes.c_int
    return fn


def test_c_quantize_q8_0_pl_matches_python(native_lib_path):
    lib = load_lib(native_lib_path)
    c_quantize = _bind_c_quantize(lib)

    rng = random.Random(0xBEEF)
    k = Q8_BLOCK * 7  # several blocks, some non-aligned in magnitude
    values = [rng.uniform(-3.0, 3.0) for _ in range(k)]
    # Force one all-zero block to exercise the amax==0 path.
    for i in range(Q8_BLOCK):
        values[2 * Q8_BLOCK + i] = 0.0

    c_values = (ctypes.c_float * k)(*values)
    c_quants = (ctypes.c_int8 * k)()
    c_scales = (ctypes.c_uint16 * (k // Q8_BLOCK))()
    rc = c_quantize(c_values, ctypes.c_uint32(k), c_quants, c_scales)
    assert rc == 0

    py_quants, py_scale_bits = matmul_q1a8._quantize_q8_0(tuple(values))
    assert list(c_quants) == py_quants
    assert list(c_scales) == py_scale_bits


def test_c_pack_matmul_q1a8_stream_matches_python(native_lib_path):
    lib = load_lib(native_lib_path)
    c_pack = _bind_c_pack(lib)

    rng = random.Random(0xC0DE)
    rows = 19  # not a multiple of 8 — exercises the partial last rowblock
    cols = 2
    k = Q1_BLOCK * 3
    rows_per_block = matmul_q1a8.ROWS_PER_BLOCK
    blocks_per_row = k // Q1_BLOCK
    weight_row_bytes = blocks_per_row * Q1_BLOCK_BYTES
    rowblocks_per_col = (rows + rows_per_block - 1) // rows_per_block
    rowblock_nbytes = matmul_q1a8._rowblock_nbytes(k, rows_per_block)
    col_stream_nbytes = rowblock_nbytes * rowblocks_per_col

    # Build random Q1_0 weights: row-major, 18 bytes per Q1 block.
    weights = bytearray(rows * weight_row_bytes)
    for i in range(len(weights)):
        weights[i] = rng.randint(0, 255)

    # Per-column int8 quants + fp16 scale bits.
    act_quants = bytearray(cols * k)
    act_scale_bits = bytearray(cols * (k // Q8_BLOCK) * 2)
    quant_list_by_col = []
    scale_list_by_col = []
    for col in range(cols):
        quants = [rng.randint(-128, 127) for _ in range(k)]
        scales = [rng.randint(0, 0xFFFF) for _ in range(k // Q8_BLOCK)]
        for i, q in enumerate(quants):
            act_quants[col * k + i] = q & 0xFF
        for i, s in enumerate(scales):
            struct.pack_into("<H", act_scale_bits, (col * (k // Q8_BLOCK) + i) * 2, s)
        quant_list_by_col.append(quants)
        scale_list_by_col.append(scales)

    c_out = (ctypes.c_uint8 * (cols * col_stream_nbytes))()
    c_weights = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
    c_quants = (ctypes.c_int8 * len(act_quants)).from_buffer(act_quants)
    c_scales = (ctypes.c_uint16 * (cols * (k // Q8_BLOCK))).from_buffer(act_scale_bits)
    rc = c_pack(c_weights, c_quants, c_scales,
                ctypes.c_uint32(rows), ctypes.c_uint32(cols), ctypes.c_uint32(k),
                c_out)
    assert rc == 0

    # Build the expected stream via the Python reference, one rowblock at a time.
    expected = bytearray(cols * col_stream_nbytes)
    py_block = bytearray(rowblock_nbytes)
    for col in range(cols):
        for rb in range(rowblocks_per_col):
            row_start = rb * rows_per_block
            row_count = min(rows_per_block, rows - row_start)
            matmul_q1a8._pack_rowblock_into(
                py_block,
                bytes(weights),
                row_start,
                row_count,
                weight_row_bytes,
                quant_list_by_col[col],
                scale_list_by_col[col],
                k,
                rows_per_block,
            )
            off = col * col_stream_nbytes + rb * rowblock_nbytes
            expected[off : off + rowblock_nbytes] = py_block

    assert bytes(c_out) == bytes(expected)


def test_bind_native_resolves_start_wait_split(native_lib_path):
    """_bind_native must resolve the run/start/wait runner entry points and
    unpack into 5 callables (the pipelining split added start_chunk/wait_chunk).
    Catches a missing C symbol or a stale binding tuple without needing a board."""
    lib = load_lib(native_lib_path)
    bound = matmul_q1a8._bind_native(lib)
    assert len(bound) == 5
    quantize, pack_acts, run_chunk, start_chunk, wait_chunk = bound
    for fn in (quantize, pack_acts, run_chunk, start_chunk, wait_chunk):
        assert callable(fn)
    # The split entry points must be distinct C functions, not the fused one.
    assert start_chunk is not run_chunk
    assert wait_chunk is not run_chunk
    assert run_chunk.restype is ctypes.c_int
    assert start_chunk.restype is ctypes.c_int
    assert wait_chunk.restype is ctypes.c_int


def test_matmul_exposes_async_scheduler_contract():
    """The PL matmul must expose run_async/complete so the graph scheduler
    pipelines it, and _PendingMatmul must carry the fields the scheduler reads
    to bar dependent ops. (The MMIO path itself is board-only.)"""
    kernel = matmul_q1a8.PLMatmulQ1A8(overlay=SimpleNamespace(
        axi_dma_0=None, axi_dma_1=None, q1a8_kernel_top_0=None))
    assert callable(getattr(kernel, "run_async", None))
    assert callable(getattr(kernel, "complete", None))
    assert kernel.backend == "pl"
    for slot in ("kernel", "op_name", "dst_handle", "dst_lo", "dst_hi"):
        assert slot in matmul_q1a8._PendingMatmul.__slots__
