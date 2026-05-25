"""Q1A8 reducer cocotb tests.

Drives random Q8 sub-blocks (32 weight bits + 32 int8 acts + 2 fp16 scales)
into the reducer and compares the registered `contribution` against a
Python golden that mirrors the truncating fp32 arithmetic bit-for-bit.

If a check fails, the assertion prints raw fp32 bits, decoded floats, and
the integer sub_sum so the offending stage is obvious.
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

CLK_PERIOD_NS = 10  # 100 MHz, matches FCLK_CLK0


# -- Python golden (bit-exact mirror of the Verilog modules) ---------------

def fp16_bits_to_fp32_bits(h: int) -> int:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return sign << 31                                  # zero / subnormal -> 0
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13)


def int_to_fp32_bits(value: int, width: int = 14) -> int:
    if value == 0:
        return 0
    mag_w = width - 1
    sign = 1 if value < 0 else 0
    mag = -value if value < 0 else value
    msb_pos = mag.bit_length() - 1                          # position of leading 1
    mag_norm = mag << (mag_w - 1 - msb_pos)
    # Drop the implicit leading 1, take the next mag_w-1 bits, right-pad to 23.
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
    prod = ((1 << 23) | ma) * ((1 << 23) | mb)              # 48-bit
    renorm = (prod >> 47) & 1
    if renorm:
        mr = (prod >> 24) & 0x7FFFFF
        er = ea + eb - 127 + 1
    else:
        mr = (prod >> 23) & 0x7FFFFF
        er = ea + eb - 127
    if er <= 0:
        return sr << 31                                     # underflow
    if er >= 255:
        return (sr << 31) | (0xFE << 23) | 0x7FFFFF         # saturate to max normal
    return (sr << 31) | (er << 23) | mr


def golden_contribution(
    weight_bits: int,
    acts: list[int],
    ws_h: int,
    as_h: int,
) -> tuple[int, int]:
    """Return (contribution_fp32_bits, sub_sum_int)."""
    sub_sum = sum(
        acts[i] if (weight_bits >> i) & 1 else -acts[i]
        for i in range(32)
    )
    ws_f32 = fp16_bits_to_fp32_bits(ws_h)
    as_f32 = fp16_bits_to_fp32_bits(as_h)
    ss_f32 = int_to_fp32_bits(sub_sum, width=14)
    combined = fp32_mul_trunc(ws_f32, as_f32)
    return fp32_mul_trunc(combined, ss_f32), sub_sum


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


async def reset(dut) -> None:
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.weight_bits.value = 0
    dut.acts_packed.value = 0
    dut.weight_scale.value = 0
    dut.act_scale.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_one(
    dut,
    weight_bits: int,
    acts: list[int],
    ws_h: int,
    as_h: int,
) -> tuple[int, int]:
    """Issue one sub-block, return (contribution_bits, valid_out)."""
    dut.weight_bits.value = weight_bits
    dut.acts_packed.value = pack_acts(acts)
    dut.weight_scale.value = ws_h
    dut.act_scale.value = as_h
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)   # output latches here
    return int(dut.contribution.value), int(dut.valid_out.value)


def _diff_msg(label, got, want, sub_sum, ws_h, as_h) -> str:
    return (
        f"{label}: "
        f"got=0x{got:08x} ({fp32_bits_to_float(got):.6g})  "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})  "
        f"sub_sum={sub_sum}  "
        f"ws=0x{ws_h:04x}  as=0x{as_h:04x}"
    )


# -- Tests -----------------------------------------------------------------

@cocotb.test()
async def test_zero(dut):
    """sub_sum=0, scales=1.0 -> contribution = 0."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset(dut)
    one = fp16_float_to_bits(1.0)
    got, valid = await drive_one(dut, 0, [0] * 32, one, one)
    assert valid == 1
    assert got == 0, f"got 0x{got:08x}"


@cocotb.test()
async def test_identity_scale(dut):
    """scales = 1.0 -> contribution is just sub_sum as fp32."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset(dut)
    one = fp16_float_to_bits(1.0)
    # Mixed signs to exercise the conditional-add tree.
    acts = [(-1) ** i * (i + 1) for i in range(32)]
    weight_bits = 0xAAAA_AAAA      # alternating bits
    got, _ = await drive_one(dut, weight_bits, acts, one, one)
    want, sub_sum = golden_contribution(weight_bits, acts, one, one)
    expected_int = int_to_fp32_bits(sub_sum)
    assert got == expected_int == want, _diff_msg("identity", got, want, sub_sum, one, one)


@cocotb.test()
async def test_corners(dut):
    """Hit the extreme sub_sums on both signs."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset(dut)
    one = fp16_float_to_bits(1.0)

    cases = [
        # name,           weight_bits,   acts,              expected_sub_sum
        ("max_pos",       0x00000000,    [-128] * 32,        +4096),
        ("max_neg",       0xFFFFFFFF,    [-128] * 32,        -4096),
        ("near_max_pos",  0xFFFFFFFF,    [+127] * 32,        +32 * 127),
        ("near_max_neg",  0x00000000,    [+127] * 32,        -32 * 127),
        ("alternating",   0x55555555,    list(range(32)),    None),
    ]
    for name, wb, acts, expected_ss in cases:
        got, _ = await drive_one(dut, wb, acts, one, one)
        want, sub_sum = golden_contribution(wb, acts, one, one)
        if expected_ss is not None:
            assert sub_sum == expected_ss, f"{name}: golden sub_sum={sub_sum} expected={expected_ss}"
        assert got == want, _diff_msg(name, got, want, sub_sum, one, one)


@cocotb.test()
async def test_random(dut):
    """1000 random Q8 sub-blocks against the bit-exact golden."""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())
    await reset(dut)
    random.seed(0xC0CCBEEF)

    failures: list[str] = []
    trials = 1000
    for trial in range(trials):
        weight_bits = random.randint(0, 2**32 - 1)
        acts = [random.randint(-128, 127) for _ in range(32)]
        # Realistic quantization scale range: amax / 127 for amax in roughly
        # [0.01, 10.0] -> scales in [~0.00008, ~0.08]. Widen to cover headroom.
        ws_h = fp16_float_to_bits(random.uniform(1e-3, 1.0))
        as_h = fp16_float_to_bits(random.uniform(1e-3, 1.0))
        got, valid = await drive_one(dut, weight_bits, acts, ws_h, as_h)
        want, sub_sum = golden_contribution(weight_bits, acts, ws_h, as_h)
        assert valid == 1
        if got != want:
            failures.append(_diff_msg(f"trial {trial}", got, want, sub_sum, ws_h, as_h))
            if len(failures) >= 5:
                break

    if failures:
        for f in failures:
            cocotb.log.error(f)
        assert False, f"{len(failures)} of {trials} trials mismatched"
