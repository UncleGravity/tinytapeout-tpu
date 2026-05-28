"""C kernel unit tests: FLASH_ATTN_EXT_F32, GET_ROWS, SET_ROWS.

These tests bypass the slab/RPC layer and exercise the C kernels directly
via ctypes against a NumPy reference. The integration with the slab
allocator and RUN_GRAPH dispatch is covered separately by the live bench.
"""

from __future__ import annotations

import ctypes
import math
import struct

import numpy as np
import pytest

# -- libbonsai_ps loader (uses the same fixture as the rest of board tests) --


@pytest.fixture(scope="module")
def lib(native_lib_path):
    return ctypes.CDLL(str(native_lib_path))


# -- helpers -----------------------------------------------------------------


def f32_to_f16_bytes(arr: np.ndarray) -> np.ndarray:
    """F32 array -> F16 bits (uint16) using NumPy's round-half-to-even.
    The C kernel uses round-toward-zero in float_to_half, so tolerances
    below are relaxed (1 ULP at fp16 is ~2^-10 of the value).
    """
    return arr.astype(np.float16).view(np.uint16)


def f16_bytes_to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.view(np.float16).astype(np.float32)


def q1_0_encode(row: np.ndarray) -> bytes:
    """Pack a F32 row of length k (k % 128 == 0) into Q1_0 wire format:
    per 128 elements: 2 bytes fp16 scale + 16 bytes (128 bits), bit=1 means
    +scale, bit=0 means -scale. Matches the C kernel's per-block dequant.
    """
    k = len(row)
    assert k % 128 == 0
    out = bytearray()
    for b in range(k // 128):
        block = row[b * 128 : (b + 1) * 128]
        amax = float(np.max(np.abs(block))) if np.any(block != 0) else 0.0
        scale = amax if amax > 0 else 1.0
        scale_f16 = np.float16(scale).view(np.uint16).item()
        out += struct.pack("<H", scale_f16)
        bits = bytearray(16)
        for i in range(128):
            v = block[i]
            # bit=1 → +scale, bit=0 → -scale. Choose nearest.
            bit = 1 if v >= 0 else 0
            bits[i // 8] |= bit << (i % 8)
        out += bytes(bits)
    return bytes(out)


def q1_0_decode_reference(packed: bytes, k: int) -> np.ndarray:
    """Inverse of q1_0_encode. Should bit-match the C kernel's dequant."""
    out = np.zeros(k, dtype=np.float32)
    for b in range(k // 128):
        blk = packed[b * 18 : (b + 1) * 18]
        scale_f16 = struct.unpack("<H", blk[:2])[0]
        scale = np.float16().view(np.uint16).__class__(scale_f16)
        scale = np.array([scale_f16], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
        for i in range(128):
            bit = (blk[2 + i // 8] >> (i % 8)) & 1
            out[b * 128 + i] = scale if bit else -scale
    return out


# ============================================================================
# GET_ROWS
# ============================================================================


def _bind_get_rows(lib, sym):
    fn = getattr(lib, sym)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
    ]
    fn.restype = ctypes.c_int
    return fn


def test_get_rows_f32_basic(lib):
    """Pick 3 rows out of a 7×8 F32 table, contiguous output."""
    fn = _bind_get_rows(lib, "bonsai_get_rows_f32")
    rng = np.random.default_rng(0)
    src = rng.standard_normal((7, 8), dtype=np.float32)
    indices = np.array([2, 5, 0], dtype=np.int32)
    dst = np.zeros((3, 8), dtype=np.float32)

    rc = fn(
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.strides[0]), ctypes.c_size_t(src.nbytes), ctypes.c_size_t(src.nbytes),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(8), ctypes.c_uint32(7),
        ctypes.c_uint32(3), ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == 0
    np.testing.assert_array_equal(dst, src[indices])


def test_get_rows_f16_basic(lib):
    fn = _bind_get_rows(lib, "bonsai_get_rows_f16")
    rng = np.random.default_rng(1)
    src_f32 = rng.standard_normal((5, 16), dtype=np.float32)
    src_f16 = src_f32.astype(np.float16)
    indices = np.array([4, 0, 2], dtype=np.int32)
    dst = np.zeros((3, 16), dtype=np.float32)

    rc = fn(
        src_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src_f16.strides[0]), ctypes.c_size_t(src_f16.nbytes), ctypes.c_size_t(src_f16.nbytes),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(16), ctypes.c_uint32(5),
        ctypes.c_uint32(3), ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == 0
    np.testing.assert_allclose(dst, src_f16[indices].astype(np.float32), rtol=0, atol=0)


def test_get_rows_q1_0_basic(lib):
    """Q1_0 row table uploaded in the packed rowblock layout."""
    from proto import q1a8_layout

    fn = _bind_get_rows(lib, "bonsai_get_rows_q1_0")
    n_rows = 10
    k = 256
    # Build rows with controlled sign + scale.
    rows_f32 = np.zeros((n_rows, k), dtype=np.float32)
    for r in range(n_rows):
        # bit pattern + scale per row
        scale = 0.5 + 0.25 * r
        signs = np.where((np.arange(k) + r) % 3 == 0, 1.0, -1.0)
        rows_f32[r] = scale * signs
    row_major = b"".join(q1_0_encode(rows_f32[r]) for r in range(n_rows))
    packed = q1a8_layout.pack_weights(row_major, n_rows, k)
    indices = np.array([9, 1, 7], dtype=np.int32)
    dst = np.zeros((3, k), dtype=np.float32)

    src0_row_bytes = (k // 128) * 18
    rc = fn(
        ctypes.c_char_p(packed),
        ctypes.c_size_t(src0_row_bytes), ctypes.c_size_t(src0_row_bytes * n_rows), ctypes.c_size_t(src0_row_bytes * n_rows),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(k), ctypes.c_uint32(n_rows),
        ctypes.c_uint32(3), ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == 0
    for i, idx in enumerate(indices):
        ref = q1_0_decode_reference(
            row_major[idx * src0_row_bytes : (idx + 1) * src0_row_bytes], k)
        np.testing.assert_array_equal(dst[i], ref)


def test_get_rows_out_of_range_index(lib):
    fn = _bind_get_rows(lib, "bonsai_get_rows_f32")
    src = np.zeros((4, 8), dtype=np.float32)
    indices = np.array([5], dtype=np.int32)  # only 4 rows
    dst = np.zeros((1, 8), dtype=np.float32)
    rc = fn(
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.strides[0]), ctypes.c_size_t(src.nbytes), ctypes.c_size_t(src.nbytes),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(8), ctypes.c_uint32(4),
        ctypes.c_uint32(1), ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == -2  # out-of-range index


# ============================================================================
# SET_ROWS
# ============================================================================


def _bind_set_rows(lib, sym):
    fn = getattr(lib, sym)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32,
    ]
    fn.restype = ctypes.c_int
    return fn


def test_set_rows_f32_to_f16_i32(lib):
    """Write 2 F32 rows into a 6-row F16 destination at indices [4, 1]."""
    fn = _bind_set_rows(lib, "bonsai_set_rows_f32_to_f16_i32")
    rng = np.random.default_rng(3)
    head_dim = 8
    n_rows_to_write = 2
    dst_rows = 6
    src = rng.standard_normal((n_rows_to_write, head_dim), dtype=np.float32)
    indices = np.array([4, 1], dtype=np.int32)
    dst = np.full((dst_rows, head_dim), 0xBEEF, dtype=np.uint16)  # poisoned

    rc = fn(
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.strides[0]), ctypes.c_size_t(src.nbytes), ctypes.c_size_t(src.nbytes),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(head_dim),
        ctypes.c_uint32(n_rows_to_write), ctypes.c_uint32(1), ctypes.c_uint32(1),
        ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == 0

    # Rows 4 and 1 should equal src (round to F16); other rows untouched.
    dst_f32 = dst.view(np.float16).astype(np.float32)
    # Per-row tolerance: 1 ULP at F16 magnitude
    for i, row in enumerate(indices):
        np.testing.assert_allclose(
            dst_f32[row], src[i].astype(np.float16).astype(np.float32),
            rtol=2e-3, atol=2e-3,
        )
    # Poisoned rows still poisoned
    untouched = [r for r in range(dst_rows) if r not in indices.tolist()]
    for r in untouched:
        assert (dst[r] == 0xBEEF).all()


def test_set_rows_f32_to_f16_i64(lib):
    fn = _bind_set_rows(lib, "bonsai_set_rows_f32_to_f16_i64")
    src = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    indices = np.array([2], dtype=np.int64)
    dst = np.zeros((4, 4), dtype=np.uint16)

    rc = fn(
        src.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(src.strides[0]), ctypes.c_size_t(src.nbytes), ctypes.c_size_t(src.nbytes),
        indices.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(indices.nbytes), ctypes.c_size_t(indices.nbytes),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[0]), ctypes.c_size_t(dst.nbytes), ctypes.c_size_t(dst.nbytes),
        ctypes.c_uint32(4),
        ctypes.c_uint32(1), ctypes.c_uint32(1), ctypes.c_uint32(1),
        ctypes.c_uint32(1), ctypes.c_uint32(1),
    )
    assert rc == 0
    assert (dst[0] == 0).all() and (dst[1] == 0).all() and (dst[3] == 0).all()
    np.testing.assert_allclose(
        dst[2].view(np.float16).astype(np.float32),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16).astype(np.float32),
        rtol=0, atol=0,
    )


# ============================================================================
# FLASH_ATTN_EXT_F32
# ============================================================================


def _bind_flash_attn(lib):
    fn = lib.bonsai_flash_attn_ext_f32
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,   # q
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,   # k
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,   # v
        ctypes.c_void_p, ctypes.c_size_t,                    # mask
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,   # dst
        ctypes.c_uint32, ctypes.c_uint32,                    # head_dim_q/v
        ctypes.c_uint32, ctypes.c_uint32,                    # n_head, n_head_kv
        ctypes.c_uint32, ctypes.c_uint32,                    # n_kv, n_token
        ctypes.c_float,
    ]
    fn.restype = ctypes.c_int
    return fn


def flash_attn_reference(q, k, v, mask, scale):
    """NumPy reference. q: [n_token, n_head, head_dim_q] F32,
    k: [n_head_kv, n_kv, head_dim_q] F32 (already dequantized from F16),
    v: [n_head_kv, n_kv, head_dim_v] F32,
    mask: [n_token, n_kv] F32 (or None),
    returns dst[n_token, n_head, head_dim_v] F32.
    """
    n_token, n_head, head_dim_q = q.shape
    n_head_kv, n_kv, _ = k.shape
    _, _, head_dim_v = v.shape
    rk2 = n_head // n_head_kv
    out = np.zeros((n_token, n_head, head_dim_v), dtype=np.float32)
    for iq1 in range(n_token):
        for h in range(n_head):
            kv_h = h // rk2
            qv = q[iq1, h]                       # [head_dim_q]
            kk = k[kv_h]                         # [n_kv, head_dim_q]
            vv = v[kv_h]                         # [n_kv, head_dim_v]
            s = qv @ kk.T * scale                # [n_kv]
            if mask is not None:
                s = s + mask[iq1]
            s_max = float(np.max(s))
            ex = np.exp(s - s_max)
            S = float(ex.sum())
            out[iq1, h] = (ex @ vv) / S
    return out


def test_flash_attn_decode_no_gqa_no_mask(lib):
    """Single-token decode, no GQA (n_head == n_head_kv), no mask."""
    fn = _bind_flash_attn(lib)
    head_dim_q = head_dim_v = 8
    n_head = n_head_kv = 4
    n_kv = 16
    n_token = 1
    scale = 1.0 / math.sqrt(head_dim_q)

    rng = np.random.default_rng(42)
    q = rng.standard_normal((n_token, n_head, head_dim_q), dtype=np.float32)
    k_f32 = rng.standard_normal((n_head_kv, n_kv, head_dim_q), dtype=np.float32)
    v_f32 = rng.standard_normal((n_head_kv, n_kv, head_dim_v), dtype=np.float32)
    k_f16 = k_f32.astype(np.float16)
    v_f16 = v_f32.astype(np.float16)

    dst = np.zeros((n_token, n_head, head_dim_v), dtype=np.float32)

    rc = fn(
        q.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(q.strides[0]),  # q_nb1: from token to next (= n_head * head_dim_q * 4)
        ctypes.c_size_t(q.strides[1]),  # q_nb2: from head to next (= head_dim_q * 4)
        k_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(k_f16.strides[1]),  # k_nb1: from kv pos (= head_dim_q * 2)
        ctypes.c_size_t(k_f16.strides[0]),  # k_nb2: from kv head (= n_kv * head_dim_q * 2)
        v_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(v_f16.strides[1]),
        ctypes.c_size_t(v_f16.strides[0]),
        ctypes.c_void_p(0), ctypes.c_size_t(0),  # no mask
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[1]),  # dst_nb1: from head (head_dim_v * 4)
        ctypes.c_size_t(dst.strides[0]),  # dst_nb2: from token
        ctypes.c_uint32(head_dim_q), ctypes.c_uint32(head_dim_v),
        ctypes.c_uint32(n_head), ctypes.c_uint32(n_head_kv),
        ctypes.c_uint32(n_kv), ctypes.c_uint32(n_token),
        ctypes.c_float(scale),
    )
    assert rc == 0

    # Reference uses F32 K, V — to match the C kernel we feed it the
    # F16-roundtripped versions.
    k_ref = k_f16.astype(np.float32)
    v_ref = v_f16.astype(np.float32)
    ref = flash_attn_reference(q, k_ref, v_ref, mask=None, scale=scale)
    # Tolerances account for: F16 conversion of K/V (~1e-3 rel),
    # different softmax ordering (online vs batched) — ~1e-5 atol expected.
    np.testing.assert_allclose(dst, ref, rtol=2e-3, atol=2e-3)


def test_flash_attn_decode_gqa_with_mask(lib):
    """Single-token decode, GQA (rk2=2), F16 mask present."""
    fn = _bind_flash_attn(lib)
    head_dim_q = head_dim_v = 16
    n_head = 4
    n_head_kv = 2
    n_kv = 32
    n_token = 1
    scale = 1.0 / math.sqrt(head_dim_q)

    rng = np.random.default_rng(7)
    q = rng.standard_normal((n_token, n_head, head_dim_q), dtype=np.float32)
    k_f16 = rng.standard_normal((n_head_kv, n_kv, head_dim_q), dtype=np.float32).astype(np.float16)
    v_f16 = rng.standard_normal((n_head_kv, n_kv, head_dim_v), dtype=np.float32).astype(np.float16)

    # Causal-ish mask: random small floats, F16
    mask_f32 = rng.standard_normal((n_token, n_kv), dtype=np.float32) * 0.1
    mask_f16 = mask_f32.astype(np.float16)

    dst = np.zeros((n_token, n_head, head_dim_v), dtype=np.float32)

    rc = fn(
        q.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(q.strides[0]), ctypes.c_size_t(q.strides[1]),
        k_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(k_f16.strides[1]), ctypes.c_size_t(k_f16.strides[0]),
        v_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(v_f16.strides[1]), ctypes.c_size_t(v_f16.strides[0]),
        mask_f16.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(mask_f16.strides[0]),
        dst.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(dst.strides[1]), ctypes.c_size_t(dst.strides[0]),
        ctypes.c_uint32(head_dim_q), ctypes.c_uint32(head_dim_v),
        ctypes.c_uint32(n_head), ctypes.c_uint32(n_head_kv),
        ctypes.c_uint32(n_kv), ctypes.c_uint32(n_token),
        ctypes.c_float(scale),
    )
    assert rc == 0

    ref = flash_attn_reference(
        q,
        k_f16.astype(np.float32),
        v_f16.astype(np.float32),
        mask_f16.astype(np.float32),
        scale=scale,
    )
    np.testing.assert_allclose(dst, ref, rtol=3e-3, atol=3e-3)


def test_flash_attn_rejects_invalid_args(lib):
    fn = _bind_flash_attn(lib)
    rc = fn(
        ctypes.c_void_p(0), ctypes.c_size_t(0), ctypes.c_size_t(0),
        ctypes.c_void_p(0), ctypes.c_size_t(0), ctypes.c_size_t(0),
        ctypes.c_void_p(0), ctypes.c_size_t(0), ctypes.c_size_t(0),
        ctypes.c_void_p(0), ctypes.c_size_t(0),
        ctypes.c_void_p(0), ctypes.c_size_t(0), ctypes.c_size_t(0),
        ctypes.c_uint32(0), ctypes.c_uint32(0),
        ctypes.c_uint32(0), ctypes.c_uint32(0),
        ctypes.c_uint32(0), ctypes.c_uint32(0),
        ctypes.c_float(1.0),
    )
    assert rc == -1  # NULL pointer rejection
