# SPDX-FileCopyrightText: © 2026 UncleGravity
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import cocotb

from bonsai_fixture import replay_fixture
from tt_protocol import ROWS, init, load_weights, run_vector


HERE = Path(__file__).parent


def dot_ref(weights: list[int], acts: list[int], seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


@cocotb.test()
async def chip_w1a8_tile_transaction(dut):
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
async def chip_reuses_stationary_weights(dut):
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
        return

    await init(dut)
    await replay_fixture(dut, Path(os.environ["BONSAI_FULL_TENSOR_FIXTURE"]))
