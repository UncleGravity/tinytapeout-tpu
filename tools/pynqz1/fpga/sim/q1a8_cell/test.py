"""Q1A8 cell cocotb tests.

Drives N sub-blocks (one full output cell at a time) and compares the
registered fp32 accumulator against a bit-exact Python golden that mirrors
the hardware's truncating fp32 arithmetic. Same pattern as the reducer
test, with `fp32_add_trunc` added.
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

CLK_PERIOD_NS = 10  # 100 MHz


# -- Python golden ---------------------------------------------------------

def fp16_bits_to_fp32_bits(h: int) -> int:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return sign << 31
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13)


def int_to_fp32_bits(value: int, width: int = 14) -> int:
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


def fp32_mul_trunc(a: int, b: int) -> int:
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


def fp32_add_trunc(a: int, b: int) -> int:
    """Bit-exact mirror of fp32_add.v."""
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    if ea == 0:
        return b
    if eb == 0:
        return a

    a_ge_b = ea >= eb
    sign_big, exp_big, mant_big = (sa, ea, (1 << 23) | ma) if a_ge_b else (sb, eb, (1 << 23) | mb)
    sign_small, _, mant_small   = (sb, eb, (1 << 23) | mb) if a_ge_b else (sa, ea, (1 << 23) | ma)
    exp_diff = exp_big - (eb if a_ge_b else ea)

    if exp_diff > 24:
        mant_small_aligned = 0
    else:
        mant_small_aligned = mant_small >> exp_diff

    if exp_diff == 0 and mant_small_aligned > mant_big:
        m1, m2, result_sign = mant_small_aligned, mant_big, sign_small
    else:
        m1, m2, result_sign = mant_big, mant_small_aligned, sign_big

    if sign_big == sign_small:
        mant_sum = m1 + m2
    else:
        mant_sum = m1 - m2

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
    mantissa = mant_norm & 0x7FFFFF
    return (result_sign << 31) | (exp_signed << 23) | mantissa


def golden_one_contribution(weight_bits, acts, ws_h, as_h):
    sub_sum = sum(acts[i] if (weight_bits >> i) & 1 else -acts[i] for i in range(32))
    ws_f32 = fp16_bits_to_fp32_bits(ws_h)
    as_f32 = fp16_bits_to_fp32_bits(as_h)
    ss_f32 = int_to_fp32_bits(sub_sum, width=14)
    combined = fp32_mul_trunc(ws_f32, as_f32)
    return fp32_mul_trunc(combined, ss_f32)


def golden_full_cell(sub_blocks) -> int:
    """Accumulate every sub-block's contribution, in order."""
    acc = 0
    for weight_bits, acts, ws_h, as_h in sub_blocks:
        c = golden_one_contribution(weight_bits, acts, ws_h, as_h)
        acc = fp32_add_trunc(acc, c)
    return acc


# -- Helpers ---------------------------------------------------------------

def pack_acts(acts: list[int]) -> int:
    packed = 0
    for i, a in enumerate(acts):
        packed |= (a & 0xFF) << (i * 8)
    return packed


def fp16_float_to_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def random_sub_block(rng: random.Random):
    weight_bits = rng.randint(0, 2**32 - 1)
    acts = [rng.randint(-128, 127) for _ in range(32)]
    ws_h = fp16_float_to_bits(rng.uniform(1e-3, 1.0))
    as_h = fp16_float_to_bits(rng.uniform(1e-3, 1.0))
    return weight_bits, acts, ws_h, as_h


async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.start_cell.value = 0
    dut.valid_in.value = 0
    dut.last_in.value = 0
    dut.weight_bits.value = 0
    dut.acts_packed.value = 0
    dut.weight_scale.value = 0
    dut.act_scale.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_cell(dut, sub_blocks, gap_cycles=0):
    """Drive one cell's worth of sub-blocks; return final acc once cell_done pulses.

    `gap_cycles`: how many idle cycles to insert between sub-blocks. 0 = back-to-back.
    """
    # Kick off
    dut.start_cell.value = 1
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    dut.start_cell.value = 0

    n = len(sub_blocks)
    for i, (wb, acts, ws_h, as_h) in enumerate(sub_blocks):
        dut.weight_bits.value = wb
        dut.acts_packed.value = pack_acts(acts)
        dut.weight_scale.value = ws_h
        dut.act_scale.value = as_h
        dut.valid_in.value = 1
        dut.last_in.value = 1 if i == n - 1 else 0
        await RisingEdge(dut.clk)

        if gap_cycles and i != n - 1:
            dut.valid_in.value = 0
            dut.last_in.value = 0
            for _ in range(gap_cycles):
                await RisingEdge(dut.clk)

    dut.valid_in.value = 0
    dut.last_in.value = 0

    # Wait for cell_done (must arrive within reducer+accumulator depth + slack)
    for _ in range(16):
        await RisingEdge(dut.clk)
        if int(dut.cell_done.value) == 1:
            assert int(dut.busy.value) == 0, "busy should drop on cell_done"
            return int(dut.acc.value)
    raise AssertionError("cell_done never fired")


def _diff_msg(label, got, want):
    return (
        f"{label}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g})  "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})"
    )


# -- Tests -----------------------------------------------------------------

@cocotb.test()
async def test_one_subblock(dut):
    """Single sub-block: acc should equal the single contribution."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xABCDEF)
    sub = random_sub_block(rng)
    got = await drive_cell(dut, [sub])
    want = golden_full_cell([sub])
    assert got == want, _diff_msg("one sub-block", got, want)


@cocotb.test()
async def test_full_k2048(dut):
    """64 sub-blocks (matches Bonsai's K=2048 inner dim)."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xC0FFEE)
    sub_blocks = [random_sub_block(rng) for _ in range(64)]
    got = await drive_cell(dut, sub_blocks)
    want = golden_full_cell(sub_blocks)
    assert got == want, _diff_msg("K=2048", got, want)


@cocotb.test()
async def test_with_gaps(dut):
    """Stream with idle cycles between sub-blocks - shouldn't change the result."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0x1357_9BDF)
    sub_blocks = [random_sub_block(rng) for _ in range(16)]
    got = await drive_cell(dut, sub_blocks, gap_cycles=3)
    want = golden_full_cell(sub_blocks)
    assert got == want, _diff_msg("gaps", got, want)


@cocotb.test()
async def test_back_to_back(dut):
    """Run several cells in succession; each must clear independently."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xFEED_FACE)
    failures = 0
    for cell in range(8):
        sub_blocks = [random_sub_block(rng) for _ in range(64)]
        got = await drive_cell(dut, sub_blocks)
        want = golden_full_cell(sub_blocks)
        if got != want:
            cocotb.log.error(_diff_msg(f"cell {cell}", got, want))
            failures += 1
    assert failures == 0, f"{failures} of 8 cells mismatched"
