import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cocotb
from cocotb.triggers import ClockCycles

from tt_protocol import (
    CMD_START,
    ROWS,
    STATUS_BUSY,
    STATUS_DONE,
    clear,
    command,
    init,
    load_weights,
    read_status,
    run_vector,
)


def dot_ref(weights: list[int], acts: list[int], seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


@cocotb.test()
async def tt_wrapper_loads_computes_and_reads_results(dut):
    await init(dut)

    weights = [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
    ]
    acts = [7, -8, -128, 5]
    seeds = [11, -13]
    expected = [dot_ref(weights[row], acts, seeds[row]) for row in range(ROWS)]

    await load_weights(dut, weights)
    got = await run_vector(dut, acts, seeds)
    assert got == expected


@cocotb.test()
async def tt_wrapper_reuses_weights_for_multiple_vectors(dut):
    await init(dut)

    weights = [
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ]
    vectors = [
        ([10, 20, -30, -40], [100, -7]),
        ([-3, 4, 5, -6], [5, 6]),
    ]

    await load_weights(dut, weights)
    for acts, seeds in vectors:
        expected = [dot_ref(weights[row], acts, seeds[row]) for row in range(ROWS)]
        got = await run_vector(dut, acts, seeds)
        assert got == expected


@cocotb.test()
async def tt_wrapper_clear_resets_control_state(dut):
    await init(dut)

    await load_weights(dut, [[1, 1, 0, 0], [0, 0, 1, 1]])
    await command(dut, CMD_START)
    await ClockCycles(dut.clk, 2)
    await clear(dut)

    status = await read_status(dut)
    assert not (status & STATUS_BUSY)
    assert not (status & STATUS_DONE)
