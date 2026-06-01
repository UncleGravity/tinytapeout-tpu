"""Each registered kernel vs. the pure-Python golden in tests/golden/kernels.py."""

from __future__ import annotations

import struct

import pytest

from board.memory.allocator import TensorAllocator
from board.profiling.timer import Timer
from proto.ops import (
    GOP_ADD_F32,
    GOP_COPY,
    GOP_GET_ROWS,
    GOP_MATMUL_Q1A8,
    GOP_MUL_F32,
    GOP_RMS_NORM_F32,
    GOP_ROPE_F32,
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
    with timer.op(op["op"]):
        kernel.run(alloc, op, timer)
    return alloc.read(op["dst"], 0, alloc.describe(op["dst"])["nbytes"])


def test_copy(registry, allocator):
    payload = b"".join(bytes([i]) for i in range(64))
    src = allocate_and_upload(allocator, payload)
    dst = allocate_empty(allocator, 64)
    out = run_one(registry, allocator, {"op": GOP_COPY, "src": src, "dst": dst, "nbytes": 64})
    assert out == payload


def test_copy_f32_to_f16(registry, allocator):
    import numpy as np

    values = np.array([0.0, 1.0, -2.5, 3.14159, 65504.0, -0.000123, 100.0, -7.0],
                      dtype=np.float32)
    src = allocate_and_upload(allocator, values.tobytes())
    dst = allocate_empty(allocator, values.size * 2)
    out = run_one(registry, allocator, {
        "op": GOP_COPY,
        "src": src,
        "dst": dst,
        "src0_type": "f32",
        "dst_type": "f16",
        "elements": int(values.size),
        "nbytes": int(values.nbytes),
    })
    got = np.frombuffer(out, dtype=np.float16)
    # The C converter is round-to-nearest-even, same as numpy's f32→f16 cast.
    assert np.array_equal(got, values.astype(np.float16))


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
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
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
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
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
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
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


def _decode_q1_row(row: bytes, k: int) -> list[float]:
    out: list[float] = []
    for block_index in range(k // Q1_BLOCK):
        block = row[
            block_index * Q1_BLOCK_BYTES : (block_index + 1) * Q1_BLOCK_BYTES
        ]
        scale = struct.unpack("<e", block[:2])[0]
        for bit in range(Q1_BLOCK):
            byte = block[2 + bit // 8]
            out.append(scale if ((byte >> (bit % 8)) & 1) else -scale)
    return out


def test_rope_f32_normal(registry, allocator):
    head_dim, n_head, n_token = 8, 4, 3
    n_dims = 8  # full rotation
    mode = 0   # NORMAL
    freq_base = 10000.0
    elements = head_dim * n_head * n_token

    src = f32_bytes(elements, seed=21)
    positions = [0, 5, 11]
    pos_bytes = struct.pack(f"<{n_token}i", *positions)

    hs = allocate_and_upload(allocator, src)
    hp = allocate_and_upload(allocator, pos_bytes)
    dst = allocate_empty(allocator, elements * F32)

    out = run_one(registry, allocator, {
        "op": GOP_ROPE_F32, "src": hs, "positions": hp, "dst": dst,
        "head_dim": head_dim, "n_head": n_head, "n_token": n_token,
        "n_dims": n_dims, "mode": mode,
        "freq_base": freq_base, "freq_scale": 1.0,
    })
    expected = golden.rope_f32(
        src, positions, head_dim, n_head, n_token, n_dims, mode, freq_base)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
        assert got[0] == pytest.approx(exp[0], abs=1e-5)


def test_rope_f32_yarn_like_bonsai(registry, allocator):
    """ROPE with YaRN params resembling Bonsai-1.7B's actual config."""
    head_dim, n_head, n_token = 128, 4, 2
    n_dims = 128
    mode = 0
    n_ctx_orig = 8192
    freq_base = 1_000_000.0
    freq_scale = 0.25
    ext_factor = 1.0
    attn_factor = 1.0
    beta_fast = 32.0
    beta_slow = 1.0
    elements = head_dim * n_head * n_token

    src = f32_bytes(elements, seed=33)
    positions = [13, 27]
    pos_bytes = struct.pack(f"<{n_token}i", *positions)

    hs = allocate_and_upload(allocator, src)
    hp = allocate_and_upload(allocator, pos_bytes)
    dst = allocate_empty(allocator, elements * F32)

    out = run_one(registry, allocator, {
        "op": GOP_ROPE_F32, "src": hs, "positions": hp, "dst": dst,
        "head_dim": head_dim, "n_head": n_head, "n_token": n_token,
        "n_dims": n_dims, "mode": mode,
        "n_ctx_orig": n_ctx_orig,
        "freq_base": freq_base, "freq_scale": freq_scale,
        "ext_factor": ext_factor, "attn_factor": attn_factor,
        "beta_fast": beta_fast, "beta_slow": beta_slow,
    })
    expected = golden.rope_f32(
        src, positions, head_dim, n_head, n_token, n_dims, mode,
        freq_base, freq_scale=freq_scale, attn_factor=attn_factor,
        ext_factor=ext_factor, beta_fast=beta_fast, beta_slow=beta_slow,
        n_ctx_orig=n_ctx_orig)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
        # YaRN ramp + freq_scale produces small numerical differences vs the
        # standard rope path; allow a bit more slack than the non-YaRN tests.
        assert got[0] == pytest.approx(exp[0], abs=1e-4)


def test_rope_f32_neox_partial(registry, allocator):
    head_dim, n_head, n_token = 16, 2, 2
    n_dims = 8  # only first 8 of 16 rotated; tail copies through
    mode = 2   # NEOX
    freq_base = 10000.0
    elements = head_dim * n_head * n_token

    src = f32_bytes(elements, seed=22)
    positions = [3, 7]
    pos_bytes = struct.pack(f"<{n_token}i", *positions)

    hs = allocate_and_upload(allocator, src)
    hp = allocate_and_upload(allocator, pos_bytes)
    dst = allocate_empty(allocator, elements * F32)

    out = run_one(registry, allocator, {
        "op": GOP_ROPE_F32, "src": hs, "positions": hp, "dst": dst,
        "head_dim": head_dim, "n_head": n_head, "n_token": n_token,
        "n_dims": n_dims, "mode": mode,
        "freq_base": freq_base, "freq_scale": 1.0,
    })
    expected = golden.rope_f32(
        src, positions, head_dim, n_head, n_token, n_dims, mode, freq_base)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
        assert got[0] == pytest.approx(exp[0], abs=1e-5)


def test_get_rows_q1_0_from_packed_layout(registry, allocator):
    from proto import q1a8_layout

    rows, k = 10, 256
    indices = [9, 1, 7]
    row_bytes = (k // Q1_BLOCK) * Q1_BLOCK_BYTES
    weights_q1_0 = _make_q1_weights(rows, k)
    weights_packed = q1a8_layout.pack_weights(weights_q1_0, rows, k)
    idx_bytes = struct.pack(f"<{len(indices)}i", *indices)

    hw = allocate_and_upload(allocator, weights_packed)
    hi = allocate_and_upload(allocator, idx_bytes)
    dst = allocate_empty(allocator, len(indices) * k * F32)

    out = run_one(registry, allocator, {
        "op": GOP_GET_ROWS,
        "src0": hw,
        "indices": hi,
        "dst": dst,
        "src0_type": "q1_0",
        "indices_type": "i32",
        "head_dim": k,
        "ne01": rows,
        "ne02": 1,
        "ne03": 1,
        "ne10": len(indices),
        "ne11": 1,
        "ne12": 1,
        "src0_nb1": row_bytes,
        "src0_nb2": row_bytes * rows,
        "src0_nb3": row_bytes * rows,
        "indices_nb1": len(idx_bytes),
        "indices_nb2": len(idx_bytes),
        "dst_nb1": k * F32,
        "dst_nb2": len(indices) * k * F32,
        "dst_nb3": len(indices) * k * F32,
    })

    expected: list[float] = []
    for idx in indices:
        row = weights_q1_0[idx * row_bytes : (idx + 1) * row_bytes]
        expected.extend(_decode_q1_row(row, k))

    got = [v[0] for v in struct.iter_unpack("<f", out)]
    assert got == pytest.approx(expected, abs=0.0)


def test_matmul_q1a8(registry, allocator):
    from proto import q1a8_layout

    rows, cols, k = 4, 2, Q1_BLOCK
    weights_q1_0 = _make_q1_weights(rows, k)
    acts = f32_bytes(cols * k, seed=17)
    # Mirror libggml-pynq.so: weights live on-board in AXIS-packed layout.
    weights_packed = q1a8_layout.pack_weights(weights_q1_0, rows, k)
    hw = allocate_and_upload(allocator, weights_packed)
    ha = allocate_and_upload(allocator, acts)
    dst = allocate_empty(allocator, rows * cols * F32)
    out = run_one(registry, allocator, {
        "op": GOP_MATMUL_Q1A8, "weights": hw, "acts": ha, "dst": dst,
        "rows": rows, "cols": cols, "k": k,
    })
    # Golden oracle still operates on the Q1_0 source — it's the ground truth.
    expected = golden.matmul_q1a8(weights_q1_0, acts, rows, cols, k)
    for got, exp in zip(struct.iter_unpack("<f", out), struct.iter_unpack("<f", expected), strict=False):
        assert got[0] == pytest.approx(exp[0], abs=1e-4)
