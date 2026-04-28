import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cocotb

from common import drive_and_sample, start_clock_and_reset, to_bits, to_signed


def read_act(dut) -> int:
    return to_signed(int(dut.act_out.value), int(dut.ACT_WIDTH.value))


def read_psum(dut) -> int:
    return to_signed(int(dut.psum_out.value), int(dut.PSUM_WIDTH.value))


async def init(dut):
    dut.clear.value = 0
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.act_in.value = 0
    dut.psum_in.value = 0
    dut.valid_in.value = 0
    await start_clock_and_reset(dut)


async def load_weight(dut, bit: int):
    await drive_and_sample(dut, weight_in=bit, weight_load=1, valid_in=0)
    await drive_and_sample(dut, weight_load=0)


async def clear_pipeline(dut):
    # Drive valid_in/weight_load to 0 so the next non-clear cycle starts idle.
    await drive_and_sample(dut, clear=1, weight_load=0, valid_in=0)


async def compute(dut, act: int, psum: int, valid: int = 1):
    aw = int(dut.ACT_WIDTH.value)
    pw = int(dut.PSUM_WIDTH.value)
    await drive_and_sample(
        dut,
        act_in=to_bits(act, aw),
        psum_in=to_bits(psum, pw),
        valid_in=valid,
        weight_load=0,
        clear=0,
    )


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

    aw = int(dut.ACT_WIDTH.value)
    pw = int(dut.PSUM_WIDTH.value)
    await drive_and_sample(
        dut,
        weight_in=0,
        weight_load=1,
        act_in=to_bits(9, aw),
        psum_in=to_bits(3, pw),
        valid_in=1,
        clear=0,
    )
    assert int(dut.weight_out.value) == 0
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    await compute(dut, act=9, psum=3)
    assert read_psum(dut) == -6
