"""Q1A8 kernel top cocotb tests.

Exercises the full AXI-Lite + AXI-Stream wrapper around q1a8_kernel. This
is the synthesizable top that goes into the Vivado block design, so any
glitch in the register file -> kernel glue would surface here under
fast iteration rather than during the Vivado build round-trip.

Tests:
  - register magic constants (sanity for the read path)
  - one full kernel: write num_subblocks, write start, stream beats,
    poll status until done, read result, verify against the golden
  - cycle counter is non-zero and bounded for a real kernel
  - back-to-back kernels reset cleanly
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

CLK_PERIOD_NS = 10  # 100 MHz, matches FCLK_CLK0

# Register offsets (must match q1a8_kernel_top.v).
REG_ID            = 0x00
REG_VERSION       = 0x04
REG_CTRL          = 0x08
REG_STATUS        = 0x0C
REG_NUM_SUBBLOCKS = 0x10
REG_RESULT        = 0x14
REG_CYCLES        = 0x18

CTRL_START = 1 << 0
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID      = 0xB05A_1000
EXPECTED_VERSION = 1


# -- Python golden (mirror of the Verilog truncating fp32 math) ------------

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
    wb, acts, ws_h, as_h = sb
    packed_acts = pack_acts(acts)
    return [
        wb & 0xFFFFFFFF,
        (packed_acts >>   0) & 0xFFFFFFFFFFFFFFFF,
        (packed_acts >>  64) & 0xFFFFFFFFFFFFFFFF,
        (packed_acts >> 128) & 0xFFFFFFFFFFFFFFFF,
        (packed_acts >> 192) & 0xFFFFFFFFFFFFFFFF,
        ws_h | (as_h << 16),
    ]


# -- AXI-Lite stimulus -----------------------------------------------------
# Hand-rolled because cocotbext-axi isn't in the flake. The protocol is
# simple enough (2 channels for write, 1 for read) that ~40 lines covers it.

async def axi_lite_write(dut, addr, data, strb=0xF):
    dut.s_axi_awaddr.value = addr
    dut.s_axi_wdata.value = data
    dut.s_axi_wstrb.value = strb
    dut.s_axi_awvalid.value = 1
    dut.s_axi_wvalid.value = 1
    dut.s_axi_bready.value = 0

    # Wait for AW+W to both be accepted (same cycle in our slave).
    while True:
        await ReadOnly()
        if int(dut.s_axi_awready.value) == 1 and int(dut.s_axi_wready.value) == 1:
            break
        await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)

    dut.s_axi_awvalid.value = 0
    dut.s_axi_wvalid.value = 0
    dut.s_axi_bready.value = 1

    while True:
        await ReadOnly()
        if int(dut.s_axi_bvalid.value) == 1:
            break
        await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)
    dut.s_axi_bready.value = 0


async def axi_lite_read(dut, addr):
    dut.s_axi_araddr.value = addr
    dut.s_axi_arvalid.value = 1
    dut.s_axi_rready.value = 0

    while True:
        await ReadOnly()
        if int(dut.s_axi_arready.value) == 1:
            break
        await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)
    dut.s_axi_arvalid.value = 0
    dut.s_axi_rready.value = 1

    while True:
        await ReadOnly()
        if int(dut.s_axi_rvalid.value) == 1:
            data = int(dut.s_axi_rdata.value)
            break
        await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)
    dut.s_axi_rready.value = 0
    return data


# -- AXIS stimulus ---------------------------------------------------------

async def push_beat(dut, data):
    dut.s_axis_tdata.value = data
    dut.s_axis_tvalid.value = 1
    while True:
        await ReadOnly()
        if int(dut.s_axis_tready.value) == 1:
            break
        await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)


# -- Setup -----------------------------------------------------------------

async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.s_axi_aclk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.s_axi_aresetn.value = 0
    for sig in [
        "s_axi_awaddr", "s_axi_awprot", "s_axi_awvalid",
        "s_axi_wdata", "s_axi_wstrb", "s_axi_wvalid",
        "s_axi_bready",
        "s_axi_araddr", "s_axi_arprot", "s_axi_arvalid",
        "s_axi_rready",
        "s_axis_tdata", "s_axis_tvalid",
    ]:
        getattr(dut, sig).value = 0
    await RisingEdge(dut.s_axi_aclk)
    await RisingEdge(dut.s_axi_aclk)
    dut.s_axi_aresetn.value = 1
    await RisingEdge(dut.s_axi_aclk)


async def poll_status_done(dut, max_polls=64):
    for _ in range(max_polls):
        status = await axi_lite_read(dut, REG_STATUS)
        if status & STATUS_DONE:
            return status
    raise AssertionError(f"STATUS.done never set (last={status:#010x})")


async def run_one_kernel(dut, sub_blocks):
    """Configure, kick, stream beats, poll done, return result."""
    await axi_lite_write(dut, REG_NUM_SUBBLOCKS, len(sub_blocks))
    await axi_lite_write(dut, REG_CTRL, CTRL_START)

    for sb in sub_blocks:
        for beat in pack_subblock_to_beats(sb):
            await push_beat(dut, beat)
    dut.s_axis_tvalid.value = 0

    await poll_status_done(dut)
    return await axi_lite_read(dut, REG_RESULT)


def _diff_msg(label, got, want):
    return (
        f"{label}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g})  "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})"
    )


# -- Tests -----------------------------------------------------------------

@cocotb.test()
async def test_register_constants(dut):
    """Sanity: ID and VERSION read back the magic constants."""
    await clk_setup(dut)
    await reset(dut)

    got_id = await axi_lite_read(dut, REG_ID)
    assert got_id == EXPECTED_ID, f"ID got 0x{got_id:08x} want 0x{EXPECTED_ID:08x}"

    got_ver = await axi_lite_read(dut, REG_VERSION)
    assert got_ver == EXPECTED_VERSION, f"VERSION got 0x{got_ver:08x}"

    # Idle status: not busy, not done
    status = await axi_lite_read(dut, REG_STATUS)
    assert status == 0, f"idle status non-zero: 0x{status:08x}"


@cocotb.test()
async def test_full_kernel(dut):
    """Configure, kick, stream a K=2048 kernel, verify result matches golden."""
    await clk_setup(dut)
    await reset(dut)

    rng = random.Random(0xCAFEC0DE)
    sub_blocks = [random_sub_block(rng) for _ in range(64)]

    got = await run_one_kernel(dut, sub_blocks)
    want = golden_full_kernel(sub_blocks)
    assert got == want, _diff_msg("full kernel", got, want)


@cocotb.test()
async def test_cycles_counter(dut):
    """CYCLES should be > 0 after a kernel and within a sane range."""
    await clk_setup(dut)
    await reset(dut)

    rng = random.Random(0x12345678)
    sub_blocks = [random_sub_block(rng) for _ in range(64)]
    await run_one_kernel(dut, sub_blocks)

    cycles = await axi_lite_read(dut, REG_CYCLES)
    # Minimum is roughly 64 sub-blocks × 6 beats + cell drain. Maximum
    # should be well under 2000 (a back-of-envelope bound for the stim flow).
    assert 64 * 6 <= cycles <= 4000, f"cycles={cycles} outside [384, 4000]"


@cocotb.test()
async def test_back_to_back(dut):
    """Three kernels in sequence; each must produce its own correct result."""
    await clk_setup(dut)
    await reset(dut)

    rng = random.Random(0xDEADC0DE)
    for i in range(3):
        sub_blocks = [random_sub_block(rng) for _ in range(64)]
        got = await run_one_kernel(dut, sub_blocks)
        want = golden_full_kernel(sub_blocks)
        assert got == want, _diff_msg(f"kernel {i}", got, want)


@cocotb.test()
async def test_varied_lengths(dut):
    """Different num_subblocks values via the register interface."""
    await clk_setup(dut)
    await reset(dut)
    rng = random.Random(0x99999)
    for n in (1, 16, 64):
        sub_blocks = [random_sub_block(rng) for _ in range(n)]
        got = await run_one_kernel(dut, sub_blocks)
        want = golden_full_kernel(sub_blocks)
        assert got == want, _diff_msg(f"len={n}", got, want)
