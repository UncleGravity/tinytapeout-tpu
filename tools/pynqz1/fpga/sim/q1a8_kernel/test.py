"""Rowblock Q1A8 kernel cocotb tests.

These tests replace the old single-cell stream tests. The stream now carries
one activation sub-block shared across several output rows.
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

CLK_PERIOD_NS = 10
ROWS = 8
Q1_BLOCK = 128
Q8_BLOCK = 32


def fp16_bits_to_fp32_bits(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return sign << 31
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13)


def int_to_fp32_bits(value, width=14):
    if value == 0:
        return 0
    mag_w = width - 1
    sign = 1 if value < 0 else 0
    mag = -value if value < 0 else value
    msb_pos = mag.bit_length() - 1
    mag_norm = mag << (mag_w - 1 - msb_pos)
    mant_top = mag_norm & ((1 << (mag_w - 1)) - 1)
    mantissa = mant_top << (23 - (mag_w - 1))
    exponent = 127 + msb_pos
    return (sign << 31) | (exponent << 23) | mantissa


def fp32_mul_trunc(a, b):
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    if ea == 0 or eb == 0:
        return 0
    sr = sa ^ sb
    prod = ((1 << 23) | ma) * ((1 << 23) | mb)
    renorm = (prod >> 47) & 1
    if renorm:
        mr = (prod >> 24) & 0x7FFFFF
        er = ea + eb - 127 + 1
    else:
        mr = (prod >> 23) & 0x7FFFFF
        er = ea + eb - 127
    if er <= 0:
        return sr << 31
    if er >= 255:
        return (sr << 31) | (0xFE << 23) | 0x7FFFFF
    return (sr << 31) | (er << 23) | mr


def fp32_add_trunc(a, b):
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    if ea == 0:
        return b
    if eb == 0:
        return a
    a_ge_b = ea >= eb
    sign_big, exp_big, mant_big = (sa, ea, (1 << 23) | ma) if a_ge_b else (sb, eb, (1 << 23) | mb)
    sign_small, _, mant_small = (sb, eb, (1 << 23) | mb) if a_ge_b else (sa, ea, (1 << 23) | ma)
    exp_diff = exp_big - (eb if a_ge_b else ea)
    mant_small_aligned = 0 if exp_diff > 24 else (mant_small >> exp_diff)
    if exp_diff == 0 and mant_small_aligned > mant_big:
        m1, m2, result_sign = mant_small_aligned, mant_big, sign_small
    else:
        m1, m2, result_sign = mant_big, mant_small_aligned, sign_big
    mant_sum = m1 + m2 if sign_big == sign_small else m1 - m2
    if mant_sum == 0:
        return 0
    lead_pos = mant_sum.bit_length() - 1
    if lead_pos > 23:
        right = lead_pos - 23
        mant_norm = mant_sum >> right
        exp_signed = exp_big + right
    elif lead_pos < 23:
        left = 23 - lead_pos
        mant_norm = mant_sum << left
        exp_signed = exp_big - left
    else:
        mant_norm = mant_sum
        exp_signed = exp_big
    if exp_signed <= 0:
        return result_sign << 31
    if exp_signed >= 255:
        return (result_sign << 31) | (0xFE << 23) | 0x7FFFFF
    return (result_sign << 31) | (exp_signed << 23) | (mant_norm & 0x7FFFFF)


def fp16_float_to_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def contribution(weight_bits, acts, ws_h, as_h):
    sub_sum = sum(acts[i] if (weight_bits >> i) & 1 else -acts[i] for i in range(Q8_BLOCK))
    ws_f32 = fp16_bits_to_fp32_bits(ws_h)
    as_f32 = fp16_bits_to_fp32_bits(as_h)
    ss_f32 = int_to_fp32_bits(sub_sum, width=14)
    return fp32_mul_trunc(fp32_mul_trunc(ws_f32, as_f32), ss_f32)


def golden_rows(weight_rows, acts, act_scales, row_count):
    out = []
    for row in range(row_count):
        acc = 0
        for q1_index, (ws_h, bits128) in enumerate(weight_rows[row]):
            for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
                q8_index = q1_index * 4 + q8_local // Q8_BLOCK
                act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
                wb = (bits128 >> q8_local) & 0xFFFF_FFFF
                acc = fp32_add_trunc(acc, contribution(wb, act_block, ws_h, act_scales[q8_index]))
        out.append(acc)
    return out


def random_case(seed, row_count=ROWS, q1_blocks=4):
    rng = random.Random(seed)
    weight_rows = []
    for row in range(row_count):
        blocks = []
        for q1_index in range(q1_blocks):
            ws_h = fp16_float_to_bits(0.01 + 0.003 * ((row + q1_index) % 17))
            bits128 = rng.getrandbits(Q1_BLOCK)
            blocks.append((ws_h, bits128))
        weight_rows.append(blocks)

    acts = [rng.randint(-128, 127) for _ in range(q1_blocks * Q1_BLOCK)]
    act_scales = [
        fp16_float_to_bits(0.001 + 0.0007 * ((i * 5 + 3) % 19))
        for i in range(q1_blocks * 4)
    ]
    return weight_rows, acts, act_scales


def pack_stream(weight_rows, acts, act_scales, row_count, q1_blocks):
    beats = []
    scale_beats = (ROWS + 3) // 4
    wbits_beats = (ROWS + 1) // 2

    for q1_index in range(q1_blocks):
        for beat in range(scale_beats):
            word = 0
            for local in range(4):
                lane = beat * 4 + local
                scale = weight_rows[lane][q1_index][0] if lane < row_count else 0
                word |= scale << (local * 16)
            beats.append(word)

        for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
            q8_index = q1_index * 4 + q8_local // Q8_BLOCK
            act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
            for beat in range(4):
                word = 0
                for byte in range(8):
                    word |= (act_block[beat * 8 + byte] & 0xFF) << (byte * 8)
                beats.append(word)
            beats.append(act_scales[q8_index])

            for beat in range(wbits_beats):
                word = 0
                for local in range(2):
                    lane = beat * 2 + local
                    bits = 0
                    if lane < row_count:
                        bits = (weight_rows[lane][q1_index][1] >> q8_local) & 0xFFFF_FFFF
                    word |= bits << (local * 32)
                beats.append(word)

    return beats


async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.start_kernel.value = 0
    dut.num_q1_blocks.value = 0
    dut.row_count.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_beat(dut, data, stall_cycles=0):
    dut.s_axis_tvalid.value = 0
    for _ in range(stall_cycles):
        await RisingEdge(dut.clk)

    dut.s_axis_tdata.value = data
    dut.s_axis_tvalid.value = 1
    while True:
        await ReadOnly()
        if int(dut.s_axis_tready.value) == 1:
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def run_kernel(dut, beats, q1_blocks, row_count, stall_rng=None):
    dut.num_q1_blocks.value = q1_blocks
    dut.row_count.value = row_count
    dut.start_kernel.value = 1
    await RisingEdge(dut.clk)
    dut.start_kernel.value = 0

    for beat in beats:
        stall = stall_rng.randint(0, 2) if stall_rng else 0
        await push_beat(dut, beat, stall_cycles=stall)
    dut.s_axis_tvalid.value = 0

    for _ in range(64):
        await RisingEdge(dut.clk)
        if int(dut.kernel_done.value) == 1:
            raw = int(dut.results_flat.value)
            return [(raw >> (lane * 32)) & 0xFFFF_FFFF for lane in range(row_count)]
    raise AssertionError("kernel_done never fired")


def diff_msg(label, got, want):
    return (
        f"{label}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g}) "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})"
    )


@cocotb.test()
async def test_one_q1_full_rowblock(dut):
    await clk_setup(dut)
    await reset(dut)
    row_count = ROWS
    q1_blocks = 1
    weight_rows, acts, act_scales = random_case(0xA11, row_count, q1_blocks)
    beats = pack_stream(weight_rows, acts, act_scales, row_count, q1_blocks)
    got = await run_kernel(dut, beats, q1_blocks, row_count)
    want = golden_rows(weight_rows, acts, act_scales, row_count)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)


@cocotb.test()
async def test_k2048_partial_rowblock(dut):
    await clk_setup(dut)
    await reset(dut)
    row_count = 5
    q1_blocks = 16
    weight_rows, acts, act_scales = random_case(0xB22, row_count, q1_blocks)
    beats = pack_stream(weight_rows, acts, act_scales, row_count, q1_blocks)
    got = await run_kernel(dut, beats, q1_blocks, row_count)
    want = golden_rows(weight_rows, acts, act_scales, row_count)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)


@cocotb.test()
async def test_k2048_with_stream_stalls(dut):
    await clk_setup(dut)
    await reset(dut)
    row_count = ROWS
    q1_blocks = 16
    weight_rows, acts, act_scales = random_case(0xC33, row_count, q1_blocks)
    beats = pack_stream(weight_rows, acts, act_scales, row_count, q1_blocks)
    got = await run_kernel(dut, beats, q1_blocks, row_count, stall_rng=random.Random(0x5150))
    want = golden_rows(weight_rows, acts, act_scales, row_count)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)
