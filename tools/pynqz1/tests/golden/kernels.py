"""Pure-Python reference implementations of every PYNQ graph op.

These are *not* loaded by the daemon. They are the oracle that the C
(and, later, PL) kernel implementations are tested against.

Each function takes raw ``bytes`` for tensors and returns the F32 ``bytes``
result, mirroring the ABI that the C kernels accept. This keeps the
golden surface flat: tests pack/unpack tensors once and compare outputs.
"""

from __future__ import annotations

import math
import struct

from proto.ops import Q1_BLOCK, Q1_BLOCK_BYTES, Q8_BLOCK

F32_BYTES = 4


def silu(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def _lround(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


def quantize_q8_0(values: tuple[float, ...]) -> tuple[list[int], list[float]]:
    """Match ``bonsai_ps.c`` Q8_0 quantization, including the fp16 scale roundtrip."""
    quants = [0] * len(values)
    scales = [0.0] * ((len(values) + Q8_BLOCK - 1) // Q8_BLOCK)

    for block_index, block_start in enumerate(range(0, len(values), Q8_BLOCK)):
        block = values[block_start : block_start + Q8_BLOCK]
        amax = 0.0
        for value in block:
            if math.isfinite(value):
                amax = max(amax, abs(value))
        if amax == 0.0:
            continue

        scale = amax / 127.0
        scales[block_index] = struct.unpack("<e", struct.pack("<e", scale))[0]
        inv_scale = 1.0 / scale
        for local_index, value in enumerate(block):
            if not math.isfinite(value):
                continue
            quant = _lround(value * inv_scale)
            quants[block_start + local_index] = min(127, max(-128, quant))

    return quants, scales


def matmul_q1a8(
    weights: bytes,
    acts: bytes,
    rows: int,
    cols: int,
    k: int,
) -> bytes:
    if k <= 0 or k % Q1_BLOCK != 0:
        raise ValueError("k must be a positive multiple of Q1_BLOCK")

    blocks_per_row = k // Q1_BLOCK
    weight_row_bytes = blocks_per_row * Q1_BLOCK_BYTES
    output = bytearray(rows * cols * F32_BYTES)

    for col in range(cols):
        act_row = struct.unpack_from(f"<{k}f", acts, col * k * F32_BYTES)
        act_quants, act_scales = quantize_q8_0(act_row)
        for row in range(rows):
            acc = 0.0
            weight_row_offset = row * weight_row_bytes
            for q1_index in range(blocks_per_row):
                block_offset = weight_row_offset + q1_index * Q1_BLOCK_BYTES
                weight_scale = struct.unpack_from("<e", weights, block_offset)[0]
                bits_offset = block_offset + struct.calcsize("<e")
                q1_base = q1_index * Q1_BLOCK
                for q8_base in range(q1_base, q1_base + Q1_BLOCK, Q8_BLOCK):
                    act_scale = act_scales[q8_base // Q8_BLOCK]
                    if weight_scale == 0.0 or act_scale == 0.0:
                        continue
                    sub_sum = 0
                    for index in range(q8_base, q8_base + Q8_BLOCK):
                        bit_index = index - q1_base
                        bit_byte = weights[bits_offset + bit_index // 8]
                        act = act_quants[index]
                        if bit_byte & (1 << (bit_index % 8)):
                            sub_sum += act
                        else:
                            sub_sum -= act
                    acc += weight_scale * act_scale * sub_sum
            struct.pack_into("<f", output, (col * rows + row) * F32_BYTES, acc)

    return bytes(output)


def _binary_f32(src0: bytes, src1: bytes, rows: int, cols: int, broadcast: bool, fn):
    elements = rows * cols
    rhs_elements = rows if broadcast else elements
    lhs = struct.unpack(f"<{elements}f", src0)
    rhs = struct.unpack(f"<{rhs_elements}f", src1)
    out = [0.0] * elements
    for col in range(cols):
        col_offset = col * rows
        for row in range(rows):
            index = col_offset + row
            rhs_index = row if broadcast else index
            out[index] = fn(lhs[index], rhs[rhs_index])
    return struct.pack(f"<{elements}f", *out)


def add_f32(src0, src1, rows, cols, broadcast=False):
    return _binary_f32(src0, src1, rows, cols, broadcast, lambda a, b: a + b)


def mul_f32(src0, src1, rows, cols, broadcast=False):
    return _binary_f32(src0, src1, rows, cols, broadcast, lambda a, b: a * b)


def scale_f32(src: bytes, elements: int, scale: float, bias: float) -> bytes:
    values = struct.unpack(f"<{elements}f", src)
    return struct.pack(f"<{elements}f", *((v * scale) + bias for v in values))


def silu_f32(src: bytes, elements: int) -> bytes:
    values = struct.unpack(f"<{elements}f", src)
    return struct.pack(f"<{elements}f", *(silu(v) for v in values))


def swiglu_f32(gate: bytes, up: bytes, elements: int) -> bytes:
    gate_values = struct.unpack(f"<{elements}f", gate)
    up_values = struct.unpack(f"<{elements}f", up)
    return struct.pack(
        f"<{elements}f",
        *(silu(g) * u for g, u in zip(gate_values, up_values, strict=False)),
    )


def _yarn_corr_dim(n_dims: int, n_ctx_orig: int, n_rot: float, base: float) -> float:
    return n_dims * math.log(n_ctx_orig / (n_rot * 2 * math.pi)) / (2 * math.log(base))


def _yarn_ramp(low: float, high: float, i0: int) -> float:
    denom = max(0.001, high - low)
    y = ((i0 / 2) - low) / denom
    return 1.0 - max(0.0, min(1.0, y))


def rope_f32(
    src: bytes,
    positions: list[int],
    head_dim: int,
    n_head: int,
    n_token: int,
    n_dims: int,
    mode: int,
    freq_base: float,
    freq_scale: float = 1.0,
    attn_factor: float = 1.0,
    ext_factor: float = 0.0,
    beta_fast: float = 0.0,
    beta_slow: float = 0.0,
    n_ctx_orig: int = 0,
) -> bytes:
    """Pure-Python reference for ROPE (NORMAL/NEOX) with optional YaRN."""
    elements = head_dim * n_head * n_token
    values = list(struct.unpack(f"<{elements}f", src))
    out = list(values)
    is_neox = (mode & 2) != 0
    use_yarn = ext_factor != 0.0

    corr_low = 0.0
    corr_high = float(n_dims)
    mscale = attn_factor
    if use_yarn:
        ctx_orig = n_ctx_orig if n_ctx_orig > 0 else 1
        start = math.floor(_yarn_corr_dim(n_dims, ctx_orig, beta_fast, freq_base))
        end = math.ceil(_yarn_corr_dim(n_dims, ctx_orig, beta_slow, freq_base))
        corr_low = max(0.0, start)
        corr_high = min(float(n_dims - 1), end)
        mscale *= 1.0 + 0.1 * math.log(1.0 / freq_scale)

    for t in range(n_token):
        pos = float(positions[t])
        for h in range(n_head):
            base = (t * n_head + h) * head_dim
            for i in range(0, n_dims, 2):
                theta_extrap = pos * (freq_base ** (-i / n_dims))
                if use_yarn:
                    theta_interp = freq_scale * theta_extrap
                    ramp_mix = _yarn_ramp(corr_low, corr_high, i) * ext_factor
                    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix
                else:
                    theta = theta_extrap * freq_scale
                c = math.cos(theta) * mscale
                s = math.sin(theta) * mscale
                if is_neox:
                    i0 = i // 2
                    i1 = i // 2 + n_dims // 2
                else:
                    i0 = i
                    i1 = i + 1
                x0 = values[base + i0]
                x1 = values[base + i1]
                out[base + i0] = x0 * c - x1 * s
                out[base + i1] = x0 * s + x1 * c
    return struct.pack(f"<{elements}f", *out)


def rms_norm_f32(src: bytes, rows: int, cols: int, eps: float) -> bytes:
    elements = rows * cols
    values = struct.unpack(f"<{elements}f", src)
    out = [0.0] * elements
    for col in range(cols):
        col_offset = col * rows
        block = values[col_offset : col_offset + rows]
        mean_square = sum(v * v for v in block) / rows
        scale = 1.0 / math.sqrt(mean_square + eps)
        for row, value in enumerate(block):
            out[col_offset + row] = value * scale
    return struct.pack(f"<{elements}f", *out)
