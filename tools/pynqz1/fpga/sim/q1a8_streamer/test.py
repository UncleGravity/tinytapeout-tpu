"""Q1A8 streamer cocotb tests.

Exercises the AXIS-style stream-driver wrapped around q1a8_cell. The
streamer is the smallest unit that will plug into the real AXI DMA flow,
so any handshake or counter bug here would be a pain to debug under
Vivado round-trips. Tests cover back-to-back streaming, stalls in the
middle of a kernel, and multiple kernels in sequence.

Same bit-exact golden as test_q1a8_cell.py - duplicated here so the sim
folder stays self-contained.
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

CLK_PERIOD_NS = 10  # 100 MHz


# -- Python golden (mirror of the Verilog truncating fp32 math) ------------

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


def golden_one_contribution(weight_bits, acts, ws_h, as_h):
    sub_sum = sum(acts[i] if (weight_bits >> i) & 1 else -acts[i] for i in range(32))
    ws_f32 = fp16_bits_to_fp32_bits(ws_h)
    as_f32 = fp16_bits_to_fp32_bits(as_h)
    ss_f32 = int_to_fp32_bits(sub_sum, width=14)
    combined = fp32_mul_trunc(ws_f32, as_f32)
    return fp32_mul_trunc(combined, ss_f32)


def golden_full_kernel(sub_blocks) -> int:
    acc = 0
    for sb in sub_blocks:
        acc = fp32_add_trunc(acc, golden_one_contribution(*sb))
    return acc


# -- Helpers ---------------------------------------------------------------

def pack_acts(acts):
    p = 0
    for i, a in enumerate(acts):
        p |= (a & 0xFF) << (i * 8)
    return p


def fp16_float_to_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def random_sub_block(rng):
    return (
        rng.randint(0, 2**32 - 1),
        [rng.randint(-128, 127) for _ in range(32)],
        fp16_float_to_bits(rng.uniform(1e-3, 1.0)),
        fp16_float_to_bits(rng.uniform(1e-3, 1.0)),
    )


async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.start_kernel.value = 0
    dut.num_subblocks.value = 0
    dut.s_valid.value = 0
    dut.s_weight_bits.value = 0
    dut.s_acts_packed.value = 0
    dut.s_weight_scale.value = 0
    dut.s_act_scale.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def _set_subblock(dut, sb):
    wb, acts, ws_h, as_h = sb
    dut.s_weight_bits.value = wb
    dut.s_acts_packed.value = pack_acts(acts)
    dut.s_weight_scale.value = ws_h
    dut.s_act_scale.value = as_h


async def push_one(dut, sb, stall_cycles=0):
    """Drive one sub-block via the AXIS handshake, with optional pre-stall.

    Stall: keep s_valid low for N cycles before raising. Exercises the
    streamer's backpressure path (cell paused, remaining unchanged).
    """
    _set_subblock(dut, sb)
    dut.s_valid.value = 0
    for _ in range(stall_cycles):
        await RisingEdge(dut.clk)

    dut.s_valid.value = 1
    # Wait for a cycle where s_ready is high at the edge. ReadOnly samples
    # the cycle's settled combinational signals just before the next edge.
    while True:
        await ReadOnly()
        if int(dut.s_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    # Transfer commits at this edge.
    await RisingEdge(dut.clk)


async def run_kernel(dut, sub_blocks, stall_rng=None):
    """Issue start_kernel, stream sub-blocks, wait for kernel_done; return result.

    If stall_rng is provided, each sub-block gets a random 0..3 stall.
    """
    dut.num_subblocks.value = len(sub_blocks)
    dut.start_kernel.value = 1
    await RisingEdge(dut.clk)
    dut.start_kernel.value = 0

    for sb in sub_blocks:
        stall = stall_rng.randint(0, 3) if stall_rng else 0
        await push_one(dut, sb, stall_cycles=stall)

    dut.s_valid.value = 0

    # Wait for kernel_done (cell drain depth is ~3 cycles).
    for _ in range(16):
        await RisingEdge(dut.clk)
        if int(dut.kernel_done.value) == 1:
            assert int(dut.busy.value) == 0, "busy should drop on kernel_done"
            return int(dut.result.value)
    raise AssertionError("kernel_done never fired")


def _diff_msg(label, got, want):
    return (
        f"{label}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g})  "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})"
    )


# -- Tests -----------------------------------------------------------------

@cocotb.test()
async def test_one_subblock(dut):
    """num_subblocks=1: minimum-length kernel."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xA1)
    sb = random_sub_block(rng)
    got = await run_kernel(dut, [sb])
    want = golden_full_kernel([sb])
    assert got == want, _diff_msg("one sub-block", got, want)


@cocotb.test()
async def test_k2048_no_stalls(dut):
    """64 sub-blocks back-to-back, s_valid never drops mid-kernel."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xB2)
    sbs = [random_sub_block(rng) for _ in range(64)]
    got = await run_kernel(dut, sbs)
    want = golden_full_kernel(sbs)
    assert got == want, _diff_msg("K=2048 no stalls", got, want)


@cocotb.test()
async def test_k2048_with_stalls(dut):
    """64 sub-blocks with random 0-3 cycle stalls between each."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xC3)
    stalls = random.Random(0xC4)
    sbs = [random_sub_block(rng) for _ in range(64)]
    got = await run_kernel(dut, sbs, stall_rng=stalls)
    want = golden_full_kernel(sbs)
    assert got == want, _diff_msg("K=2048 with stalls", got, want)


@cocotb.test()
async def test_multiple_kernels(dut):
    """Five kernels in sequence, each must independently clear and finish."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xD5)
    for i in range(5):
        sbs = [random_sub_block(rng) for _ in range(64)]
        got = await run_kernel(dut, sbs)
        want = golden_full_kernel(sbs)
        assert got == want, _diff_msg(f"kernel {i}", got, want)


@cocotb.test()
async def test_varied_lengths(dut):
    """Different num_subblocks values - boundary conditions for `remaining`."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xE6)
    for n in (1, 2, 16, 33, 64, 100):
        sbs = [random_sub_block(rng) for _ in range(n)]
        got = await run_kernel(dut, sbs)
        want = golden_full_kernel(sbs)
        assert got == want, _diff_msg(f"len={n}", got, want)
