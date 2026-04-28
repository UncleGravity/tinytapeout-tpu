import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cocotb

from common import (
    dot_ref,
    drive_and_sample,
    make_acts,
    make_weights,
    pack_lanes,
    start_clock_and_reset,
    to_bits,
    to_signed,
    unpack_lanes,
)


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


async def cycle(dut, lanes=None, seed: int = 0, valid: int = 0,
                weight_load: int = 0, weight_in: int = 0):
    cols = int(dut.COLS.value)
    aw = int(dut.ACT_WIDTH.value)
    pw = int(dut.PSUM_WIDTH.value)
    if lanes is None:
        lanes = [0] * cols
    await drive_and_sample(
        dut,
        clear=0,
        weight_load=weight_load,
        weight_in=weight_in,
        act_in=pack_lanes(lanes, aw),
        psum_in=to_bits(seed, pw),
        valid_in=valid,
    )


async def load_weights(dut, weights):
    # Serial chain enters at column 0; load last column first so PE order matches.
    for bit in reversed(weights):
        await cycle(dut, weight_load=1, weight_in=bit)
    await cycle(dut)


async def clear_row(dut):
    await drive_and_sample(dut, clear=1, weight_load=0, valid_in=0)
    await drive_and_sample(dut, clear=0)


async def drive_one_dot(dut, acts, seed: int = 0):
    cols = int(dut.COLS.value)
    for step in range(cols):
        lanes = [0] * cols
        lanes[step] = acts[step]
        await cycle(
            dut,
            lanes=lanes,
            seed=seed if step == 0 else 0,
            valid=1 if step == 0 else 0,
        )


@cocotb.test()
async def loads_weights_through_serial_chain(dut):
    await init(dut)
    cols = int(dut.COLS.value)

    weights = make_weights(1, cols, kind=0)[0]
    await load_weights(dut, weights)

    acts = make_acts(cols, kind=0)
    await drive_one_dot(dut, acts)

    assert int(dut.valid_out.value) == 1
    expected = dot_ref(weights, acts)
    got = read_psum(dut)
    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def computes_seeded_systolic_dot_product(dut):
    await init(dut)
    cols = int(dut.COLS.value)

    weights = make_weights(1, cols, kind=1)[0]
    acts = make_acts(cols, kind=1)
    seed = -11

    await load_weights(dut, weights)
    await drive_one_dot(dut, acts, seed=seed)

    assert int(dut.valid_out.value) == 1
    expected = dot_ref(weights, acts, seed)
    got = read_psum(dut)
    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def supports_back_to_back_wavefronts(dut):
    await init(dut)
    cols = int(dut.COLS.value)

    weights = make_weights(1, cols, kind=2)[0]
    acts_by_op = [make_acts(cols, kind=k + 3) for k in range(2)]
    seeds = [100, -7]
    expected = [
        dot_ref(weights, acts_by_op[op], seeds[op])
        for op in range(len(acts_by_op))
    ]

    await load_weights(dut, weights)

    got = []
    for step in range(cols + len(acts_by_op) - 1):
        lanes = [0] * cols
        for col in range(cols):
            op = step - col
            if 0 <= op < len(acts_by_op):
                lanes[col] = acts_by_op[op][col]

        valid = 1 if step < len(acts_by_op) else 0
        seed = seeds[step] if step < len(seeds) else 0
        await cycle(dut, lanes=lanes, seed=seed, valid=valid)

        if int(dut.valid_out.value):
            got.append(read_psum(dut))

    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def forwards_activation_lanes_to_next_row(dut):
    await init(dut)
    cols = int(dut.COLS.value)
    aw = int(dut.ACT_WIDTH.value)

    await load_weights(dut, [1] * cols)

    lanes = make_acts(cols, kind=4)
    await cycle(dut, lanes=lanes, valid=0)
    assert unpack_lanes(int(dut.act_out.value), cols, aw) == lanes


@cocotb.test()
async def clear_flushes_pipeline_but_not_weights(dut):
    await init(dut)
    cols = int(dut.COLS.value)

    weights = make_weights(1, cols, kind=5)[0]
    await load_weights(dut, weights)
    await drive_one_dot(dut, make_acts(cols, kind=5))
    assert int(dut.valid_out.value) == 1

    await clear_row(dut)
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0

    acts = make_acts(cols, kind=6)
    await drive_one_dot(dut, acts)
    expected = dot_ref(weights, acts)
    got = read_psum(dut)
    assert got == expected, f"got={got} expected={expected}"
