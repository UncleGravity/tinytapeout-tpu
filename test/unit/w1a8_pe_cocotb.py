import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge


ACT_WIDTH = 8
PSUM_WIDTH = 16


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


def read_act(dut) -> int:
    return to_signed(int(dut.act_out.value), ACT_WIDTH)


def read_psum(dut) -> int:
    return to_signed(int(dut.psum_out.value), PSUM_WIDTH)


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.act_in.value = 0
    dut.psum_in.value = 0
    dut.valid_in.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def load_weight(dut, bit: int):
    await FallingEdge(dut.clk)
    dut.weight_in.value = bit
    dut.weight_load.value = 1
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.weight_load.value = 0


async def clear_pipeline(dut):
    await FallingEdge(dut.clk)
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.clear.value = 0


async def compute(dut, act: int, psum: int, valid: int = 1):
    await FallingEdge(dut.clk)
    dut.act_in.value = to_bits(act, ACT_WIDTH)
    dut.psum_in.value = to_bits(psum, PSUM_WIDTH)
    dut.valid_in.value = valid
    dut.weight_load.value = 0
    dut.clear.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.valid_in.value = 0


@cocotb.test()
async def reset_and_clear_preserve_loaded_weight(dut):
    await init(dut)

    assert int(dut.weight_out.value) == 0
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    await load_weight(dut, 1)
    assert int(dut.weight_out.value) == 1

    await compute(dut, act=5, psum=10)
    assert int(dut.valid_out.value) == 1
    assert read_act(dut) == 5
    assert read_psum(dut) == 15

    await clear_pipeline(dut)
    assert int(dut.weight_out.value) == 1
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    await compute(dut, act=7, psum=0)
    assert read_psum(dut) == 7


@cocotb.test()
async def add_subtract_and_int8_min_are_exact(dut):
    await init(dut)

    await load_weight(dut, 1)
    await compute(dut, act=127, psum=-3)
    assert read_psum(dut) == 124

    await load_weight(dut, 0)
    await compute(dut, act=5, psum=10)
    assert read_psum(dut) == 5

    await compute(dut, act=-128, psum=-1)
    assert read_psum(dut) == 127


@cocotb.test()
async def invalid_cycle_creates_bubble(dut):
    await init(dut)
    await load_weight(dut, 1)

    await compute(dut, act=50, psum=100, valid=0)
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    await compute(dut, act=50, psum=100, valid=1)
    assert int(dut.valid_out.value) == 1
    assert read_psum(dut) == 150


@cocotb.test()
async def weight_load_has_priority_over_compute(dut):
    await init(dut)
    await load_weight(dut, 1)

    await FallingEdge(dut.clk)
    dut.weight_in.value = 0
    dut.weight_load.value = 1
    dut.act_in.value = to_bits(9, ACT_WIDTH)
    dut.psum_in.value = to_bits(3, PSUM_WIDTH)
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.weight_load.value = 0
    dut.valid_in.value = 0

    assert int(dut.weight_out.value) == 0
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    await compute(dut, act=9, psum=3)
    assert read_psum(dut) == -6
