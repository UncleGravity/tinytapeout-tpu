"""Q1A8 kernel end-to-end cocotb tests.

Drives the kernel through its 64-bit AXIS data port (the same shape the
real AXI DMA will use) and compares the result against the bit-exact
golden. End-to-end: AXIS beat packer -> streamer -> cell -> reducer -> fp32 math.

If a test fails:
- one_subblock fails -> bug in axis_to_subblock for the basic 6-beat case
- k2048 fails        -> bug in either packer multi-sub-block handling, or
                        an interaction with the streamer's last_in timing
- with_stalls fails  -> beat-level backpressure path in axis_to_subblock
- multiple_kernels   -> packer or streamer state not cleanly reset
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

CLK_PERIOD_NS = 10  # 100 MHz


# -- Golden ----------------------------------------------------------------

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


def golden_full_kernel(sub_blocks):
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


def pack_subblock_to_beats(sb):
    """Lay out one sub-block as six 64-bit beats per axis_to_subblock.v."""
    wb, acts, ws_h, as_h = sb
    packed_acts = pack_acts(acts)
    return [
        wb & 0xFFFFFFFF,                                  # beat 0
        (packed_acts >>   0) & 0xFFFFFFFFFFFFFFFF,        # beat 1
        (packed_acts >>  64) & 0xFFFFFFFFFFFFFFFF,        # beat 2
        (packed_acts >> 128) & 0xFFFFFFFFFFFFFFFF,        # beat 3
        (packed_acts >> 192) & 0xFFFFFFFFFFFFFFFF,        # beat 4
        ws_h | (as_h << 16),                              # beat 5
    ]


async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.start_kernel.value = 0
    dut.num_subblocks.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_beat(dut, data, stall_cycles=0):
    """Drive one AXIS beat, with optional pre-stall to test backpressure."""
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


async def run_kernel(dut, sub_blocks, stall_rng=None):
    dut.num_subblocks.value = len(sub_blocks)
    dut.start_kernel.value = 1
    await RisingEdge(dut.clk)
    dut.start_kernel.value = 0

    for sb in sub_blocks:
        for beat in pack_subblock_to_beats(sb):
            stall = stall_rng.randint(0, 2) if stall_rng else 0
            await push_beat(dut, beat, stall_cycles=stall)

    dut.s_axis_tvalid.value = 0

    # Drain depth: cell pipeline (3 cycles) + packer hold cycle + slack.
    for _ in range(32):
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
    """Six beats -> one sub-block -> one fp32 result."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xA1)
    sb = random_sub_block(rng)
    got = await run_kernel(dut, [sb])
    want = golden_full_kernel([sb])
    assert got == want, _diff_msg("one sub-block", got, want)


@cocotb.test()
async def test_k2048(dut):
    """64 sub-blocks (= 384 beats), no AXIS stalls. Full K=2048 inner dim."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xB2)
    sbs = [random_sub_block(rng) for _ in range(64)]
    got = await run_kernel(dut, sbs)
    want = golden_full_kernel(sbs)
    assert got == want, _diff_msg("K=2048", got, want)


@cocotb.test()
async def test_k2048_with_beat_stalls(dut):
    """64 sub-blocks with random 0-2 cycle stalls per beat - exercises the
    packer's beat-level handshake under bursty DMA."""
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
    """Three kernels in sequence; packer + streamer state must clear between."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xD5)
    for i in range(3):
        sbs = [random_sub_block(rng) for _ in range(64)]
        got = await run_kernel(dut, sbs)
        want = golden_full_kernel(sbs)
        assert got == want, _diff_msg(f"kernel {i}", got, want)


@cocotb.test()
async def test_varied_lengths(dut):
    """Sub-block-count edge cases."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0xE6)
    for n in (1, 2, 16, 33, 64):
        sbs = [random_sub_block(rng) for _ in range(n)]
        got = await run_kernel(dut, sbs)
        want = golden_full_kernel(sbs)
        assert got == want, _diff_msg(f"len={n}", got, want)
