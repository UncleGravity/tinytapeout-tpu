# SPDX-FileCopyrightText: © 2026 UncleGravity
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import cocotb
from cocotb.triggers import ClockCycles

from bonsai_fixture import replay_fixture
from common import dot_ref, make_acts, make_seeds, make_weights
from tt_protocol import (
    CMD_START,
    STATUS_BUSY,
    STATUS_DONE,
    clear,
    command,
    init,
    load_weights,
    read_status,
    run_vector,
    tile,
)


HERE = Path(__file__).parent


@cocotb.test()
async def chip_loads_computes_and_reads_results(dut):
    await init(dut)
    t = tile(dut)

    weights = make_weights(t.rows, t.cols, kind=0)
    acts    = make_acts(t.cols, kind=0)
    seeds   = make_seeds(t.rows, kind=0)
    expected = [dot_ref(weights[r], acts, seeds[r]) for r in range(t.rows)]

    await load_weights(dut, weights)
    got = await run_vector(dut, acts, seeds)
    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def chip_reuses_stationary_weights(dut):
    await init(dut)
    t = tile(dut)

    weights = make_weights(t.rows, t.cols, kind=1)
    vectors = [
        (make_acts(t.cols, kind=k), make_seeds(t.rows, kind=k))
        for k in range(1, 4)
    ]

    await load_weights(dut, weights)
    for acts, seeds in vectors:
        expected = [dot_ref(weights[r], acts, seeds[r]) for r in range(t.rows)]
        got = await run_vector(dut, acts, seeds)
        assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def chip_clear_resets_control_state(dut):
    await init(dut)
    t = tile(dut)

    await load_weights(dut, make_weights(t.rows, t.cols, kind=2))
    await command(dut, CMD_START)
    await ClockCycles(dut.clk, 2)
    await clear(dut)

    status = await read_status(dut)
    assert not (status & STATUS_BUSY), f"BUSY still set: status=0x{status:02x}"
    assert not (status & STATUS_DONE), f"DONE still set: status=0x{status:02x}"


@cocotb.test()
async def chip_matches_bonsai_q1_0_group_fixtures(dut):
    await init(dut)

    fixture_paths = [
        HERE / "fixtures" / "bonsai_blk0_attn_q_r0_r1_g0.json",
        HERE / "fixtures" / "bonsai_blk0_attn_q_r42_r43_g7.json",
    ]
    for fixture_path in fixture_paths:
        await replay_fixture(dut, fixture_path)


@cocotb.test()
async def chip_matches_bonsai_q1_0_full_row_tile_fixture(dut):
    await init(dut)
    await replay_fixture(dut, HERE / "fixtures" / "bonsai_blk0_attn_q_rows0_1_all_groups.json")


@cocotb.test()
async def chip_matches_bonsai_q1_0_full_tensor_fixture(dut):
    if os.getenv("BONSAI_FULL_TENSOR_TESTS") != "1":
        cocotb.log.warning(
            "skipping chip_matches_bonsai_q1_0_full_tensor_fixture: "
            "set BONSAI_FULL_TENSOR_TESTS=1 to enable"
        )
        return

    await init(dut)
    await replay_fixture(dut, Path(os.environ["BONSAI_FULL_TENSOR_FIXTURE"]))
