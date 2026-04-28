"""Shared cocotb helpers used by chip-level and per-module tests.

Lives at test/common.py; test/unit/ adds test/ to sys.path so it imports the
same module from either scope.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


BITS_PER_BYTE = 8


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


def pack_lanes(values, width: int) -> int:
    packed = 0
    for i, v in enumerate(values):
        packed |= to_bits(v, width) << (i * width)
    return packed


def unpack_lanes(raw: int, count: int, width: int) -> list[int]:
    return [to_signed(raw >> (i * width), width) for i in range(count)]


def dot_ref(weights, acts, seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


async def start_clock_and_reset(dut, period_ns: int = 10):
    """Start a clock on dut.clk and pulse rst_n low for a few cycles. The
    caller is responsible for any other input defaults before calling."""
    cocotb.start_soon(Clock(dut.clk, period_ns, unit="ns").start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def drive_and_sample(dut, **signals):
    """Drive named signals on the falling edge, then resync after the next
    rising edge so registered outputs are stable to read."""
    await FallingEdge(dut.clk)
    for name, value in signals.items():
        getattr(dut, name).value = value
    await RisingEdge(dut.clk)
    await ReadOnly()


# --- deterministic test data generators -------------------------------------
# Mixed-sign palettes including int8 boundaries; indexed deterministically so
# tests stay reproducible across COLS / ROWS values.

_ACT_PALETTE = [-128, 7, -8, 5, 100, -7, -3, 4, 64, -64, 1, -2, 3, -4, 10, 20, -30, -40]
# Stays clear of int16 boundary so seed + COLS * 128 does not overflow PSUM_WIDTH=16
# even for the maximum protocol-supported COLS=8.
_SEED_PALETTE = [11, -13, 100, -7, 5, 6, -3, 9, 30000, -30000, 0, 1234, -2345, 42, -42, 7777]


def make_acts(cols: int, kind: int = 0) -> list[int]:
    return [_ACT_PALETTE[(kind * 5 + i * 3) % len(_ACT_PALETTE)] for i in range(cols)]


def make_weights(rows: int, cols: int, kind: int = 0) -> list[list[int]]:
    return [
        [(kind * 11 + r * 7 + c * 5 + 1) & 1 for c in range(cols)]
        for r in range(rows)
    ]


def make_seeds(rows: int, kind: int = 0) -> list[int]:
    return [_SEED_PALETTE[(kind * 7 + r * 3) % len(_SEED_PALETTE)] for r in range(rows)]
