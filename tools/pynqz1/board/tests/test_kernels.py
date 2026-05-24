"""Each registered kernel vs. the pure-Python golden in tests/golden/kernels.py."""

from __future__ import annotations

import struct

import pytest

from board.memory.allocator import TensorAllocator
from board.profiling.timer import Timer
from proto.ops import (
    GOP_ADD_F32,
    GOP_COPY,
    GOP_MATMUL_Q1A8,
    GOP_MUL_F32,
    GOP_RMS_NORM_F32,
    GOP_SCALE_F32,
    GOP_SILU_F32,
    GOP_SWIGLU_F32,
    Q1_BLOCK,
    Q1_BLOCK_BYTES,
)
from tests.golden import kernels as golden
from tests.golden.vectors import f32_bytes


F32 = 4


def allocate_and_upload(alloc: TensorAllocator, data: bytes) -> int:
    record = alloc.allocate(len(data), shape=[len(data)], dtype="u8")
    alloc.write(record.handle, 0, data)
    return record.handle


def allocate_empty(alloc: TensorAllocator, nbytes: int) -> int:
    return alloc.allocate(nbytes, shape=[nbytes], dtype="u8").handle


def run_one(registry, alloc: TensorAllocator, op: dict) -> bytes:
    kernel = registry.get(op["op"])
    timer = Timer()
    with timer.op(op["op"], index=0):
        kernel.run(alloc, op, timer)
    return alloc.read(op["dst"], 0, alloc.describe(op["dst"])["nbytes"])


def test_copy(registry, allocator):
    payload = b"".join(bytes([i]) for i in range(64))
    src = allocate_and_upload(allocator, payload)
    dst = allocate_empty(allocator, 64)
    out = run_one(registry, allocator, {"op": GOP_COPY, "src": src, "dst": dst, "nbytes": 64})
    assert out == payload


def test_add_f32(registry, allocator):
    rows, cols = 8, 4
    src0 = f32_bytes(rows * cols, seed=1)
    src1 = f32_bytes(rows * cols, seed=11)
    h0 = allocate_and_upload(allocator, src0)
    h1 = allocate_and_upload(allocator, src1)
    dst = allocate_empty(allocator, rows * cols * F32)
    out = run_one(registry, allocator, {
        "op": GOP_ADD_F32, "src0": h0, "src1": h1, "dst": dst,
        "rows": rows, "cols": cols,
    })
    assert out == golden.add_f32(src0, src1, rows, cols)


def test_mul_f32_broadcast(registry, allocator):
    rows, cols = 8, 4
    src0 = f32_bytes(rows * cols, seed=3)
    src1 = f32_bytes(rows, seed=7)
    h0 = allocate_and_upload(allocator, src0)
    h1 = allocate_and_upload(allocator, src1)
    dst = allocate_empty(allocator, rows * cols * F32)
    out = run_one(registry, allocator, {
        "op": GOP_MUL_F32, "src0": h0, "src1": h1, "dst": dst,
        "rows": rows, "cols": cols, "src1_broadcast": True,
    })
    assert out == golden.mul_f32(src0, src1, rows, cols, broadcast=True)


def test_scale_f32(registry, allocator):
    elements = 32
    src = f32_bytes(elements, seed=5)
    h = allocate_and_upload(allocator, src)
    dst = allocate_empty(allocator, elements * F32)
    out = run_one(registry, allocator, {
        "op": GOP_SCALE_F32, "src": h, "dst": dst,
        "elements": elements, "scale": 0.25, "bias": -0.5,
    })
    assert out == golden.scale_f32(src, elements, 0.25, -0.5)


def test_silu_f32(registry, allocator):
    elements = 16
    src = f32_bytes(elements, seed=9)
    h = allocate_and_upload(allocator, src)
    dst = allocate_empty(allocator, elements * F32)
    out = run_one(registry, allocator, {
        "op": GOP_SILU_F32, "src": h, "dst": dst, "elements": elements,
    })
    expected = golden.silu_f32(src, elements)
    # F32 nonlinearity — allow tiny numerical drift.
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected)):
        assert got[0] == pytest.approx(exp[0], abs=1e-6)


def test_swiglu_f32(registry, allocator):
    elements = 16
    gate = f32_bytes(elements, seed=2)
    up = f32_bytes(elements, seed=4)
    hg = allocate_and_upload(allocator, gate)
    hu = allocate_and_upload(allocator, up)
    dst = allocate_empty(allocator, elements * F32)
    out = run_one(registry, allocator, {
        "op": GOP_SWIGLU_F32, "src0": hg, "src1": hu, "dst": dst,
        "elements": elements,
    })
    expected = golden.swiglu_f32(gate, up, elements)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected)):
        assert got[0] == pytest.approx(exp[0], abs=1e-6)


def test_rms_norm_f32(registry, allocator):
    rows, cols = 64, 2
    src = f32_bytes(rows * cols, seed=13)
    h = allocate_and_upload(allocator, src)
    dst = allocate_empty(allocator, rows * cols * F32)
    out = run_one(registry, allocator, {
        "op": GOP_RMS_NORM_F32, "src": h, "dst": dst,
        "rows": rows, "cols": cols, "eps": 1e-6,
    })
    expected = golden.rms_norm_f32(src, rows, cols, 1e-6)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected)):
        assert got[0] == pytest.approx(exp[0], abs=1e-5)


def _make_q1_weights(rows: int, k: int) -> bytes:
    # Deterministic bit pattern + a varying fp16 scale per block.
    out = bytearray()
    blocks_per_row = k // Q1_BLOCK
    for row in range(rows):
        for block_index in range(blocks_per_row):
            scale = 0.0625 * (1 + ((row + block_index) % 5))
            out += struct.pack("<e", scale)
            bits = bytearray(Q1_BLOCK // 8)
            for bit in range(Q1_BLOCK):
                if ((row * 13 + block_index * 7 + bit * 3) & 1):
                    bits[bit // 8] |= 1 << (bit % 8)
            out += bits
    assert len(out) == rows * blocks_per_row * Q1_BLOCK_BYTES
    return bytes(out)


def test_matmul_q1a8(registry, allocator):
    rows, cols, k = 4, 2, Q1_BLOCK
    weights = _make_q1_weights(rows, k)
    acts = f32_bytes(cols * k, seed=17)
    hw = allocate_and_upload(allocator, weights)
    ha = allocate_and_upload(allocator, acts)
    dst = allocate_empty(allocator, rows * cols * F32)
    out = run_one(registry, allocator, {
        "op": GOP_MATMUL_Q1A8, "weights": hw, "acts": ha, "dst": dst,
        "rows": rows, "cols": cols, "k": k,
    })
    expected = golden.matmul_q1a8(weights, acts, rows, cols, k)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected)):
        assert got[0] == pytest.approx(exp[0], abs=1e-4)
