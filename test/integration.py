# SPDX-FileCopyrightText: © 2026 UncleGravity
# SPDX-License-Identifier: Apache-2.0

import json
import os
import struct
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


HERE = Path(__file__).parent
ROWS = 2
COLS = 4
Q8_BLOCK_SIZE = 32
FLOAT_TOLERANCE = 1e-4

CMD_NOP = 0
CMD_CLEAR = 1
CMD_LOAD_WEIGHT = 2
CMD_LOAD_ACT = 3
CMD_LOAD_SEED = 4
CMD_START = 5
CMD_READ_RESULT = 6
CMD_STATUS = 7

STATUS_DONE = 1 << 1
STATUS_WEIGHT_DONE = 1 << 2
STATUS_ALL_VALID = 1 << 3
STATUS_START_READY = 1 << 4
STATUS_WEIGHT_READY = 1 << 5
STATUS_ROW0_VALID = 1 << 6
STATUS_ROW1_VALID = 1 << 7


def pack_ui(cmd: int, index: int = 0, row: int = 0) -> int:
    return (cmd & 0x7) | ((index & 0x3) << 3) | ((row & 0x1) << 5)


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


def dot_ref(weights: list[int], acts: list[int], seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def command(dut, cmd: int, data: int = 0, index: int = 0, row: int = 0):
    await FallingEdge(dut.clk)
    dut.ui_in.value = pack_ui(cmd, index=index, row=row)
    dut.uio_in.value = data & 0xFF
    await RisingEdge(dut.clk)
    await ReadOnly()


async def nop(dut):
    await command(dut, CMD_NOP)


async def read_status(dut) -> int:
    await command(dut, CMD_STATUS)
    return int(dut.uo_out.value) & 0xFF


async def load_weights(dut, weights_by_row: list[list[int]]):
    # Physical shift order: send the last column first.
    for col in reversed(range(COLS)):
        bits = 0
        for row in range(ROWS):
            bits |= weights_by_row[row][col] << row
        await command(dut, CMD_LOAD_WEIGHT, data=bits)

    await nop(dut)
    status = await read_status(dut)
    assert status & STATUS_WEIGHT_DONE
    assert status & STATUS_WEIGHT_READY


async def load_acts(dut, acts: list[int]):
    for col, act in enumerate(acts):
        await command(dut, CMD_LOAD_ACT, data=to_bits(act, 8), index=col)


async def load_seed(dut, row: int, seed: int):
    raw = to_bits(seed, 24)
    for byte in range(3):
        await command(
            dut,
            CMD_LOAD_SEED,
            data=(raw >> (8 * byte)) & 0xFF,
            index=byte,
            row=row,
        )


async def start(dut):
    status = await read_status(dut)
    assert status & STATUS_START_READY
    await command(dut, CMD_START)
    await nop(dut)


async def wait_done(dut, limit: int = 64) -> int:
    for _ in range(limit):
        status = await read_status(dut)
        if status & STATUS_DONE:
            return status
    raise AssertionError("top wrapper did not report done")


async def read_result(dut, row: int) -> int:
    raw = 0
    for byte in range(3):
        await command(dut, CMD_READ_RESULT, index=byte, row=row)
        raw |= (int(dut.uo_out.value) & 0xFF) << (8 * byte)
    return to_signed(raw, 24)


async def run_vector(dut, acts: list[int], seeds: list[int]) -> list[int]:
    await load_acts(dut, acts)
    for row, seed in enumerate(seeds):
        await load_seed(dut, row, seed)
    await start(dut)
    status = await wait_done(dut)
    assert status & STATUS_ALL_VALID
    assert status & STATUS_ROW0_VALID
    assert status & STATUS_ROW1_VALID
    return [await read_result(dut, row) for row in range(ROWS)]


async def replay_fixture(dut, fixture_path: Path):
    fixture = json.loads(fixture_path.read_text())

    assert fixture["source"]["tensor"] == "blk.0.attn_q.weight"
    assert fixture["source"]["type"] == "q1_0"
    assert fixture["tile"] == {"rows": ROWS, "cols": COLS}
    assert fixture["schema_version"] == 1
    assert fixture["quantization"]["q8_0_block_size"] == Q8_BLOCK_SIZE
    assert Q8_BLOCK_SIZE % COLS == 0
    assert len(fixture["reference"]["integer_final"]) == ROWS
    assert len(fixture["reference"]["ggml_scaled_float"]) == ROWS
    assert len(fixture["quantization"]["q1_scales_fp16_hex_by_group"]) > 0
    assert len(fixture["quantization"]["q8_scales_fp16_hex_by_group"]) == len(
        fixture["quantization"]["q1_scales_fp16_hex_by_group"]
    )

    scaled_from_rtl = [0.0 for _ in range(ROWS)]
    block_sums = [0 for _ in range(ROWS)]
    current_block = None

    for txn in fixture["transactions"]:
        assert len(txn["weights"]) == ROWS
        assert all(len(row_weights) == COLS for row_weights in txn["weights"])
        assert len(txn["acts"]) == COLS
        assert len(txn["seeds"]) == ROWS
        assert len(txn["expected"]) == ROWS
        assert txn["cols"][0] % COLS == 0

        block_key = (txn["group"], (txn["cols"][0] % 128) // Q8_BLOCK_SIZE)
        if block_key != current_block:
            if current_block is not None:
                accumulate_scaled_block(fixture, current_block, block_sums, scaled_from_rtl)
            current_block = block_key
            block_sums = [0 for _ in range(ROWS)]

        deltas = [
            txn["expected"][row] - txn["seeds"][row]
            for row in range(ROWS)
        ]
        expected = [
            block_sums[row] + deltas[row]
            for row in range(ROWS)
        ]

        await load_weights(dut, txn["weights"])
        got = await run_vector(dut, txn["acts"], block_sums)
        assert got == expected
        block_sums = got

    if current_block is not None:
        accumulate_scaled_block(fixture, current_block, block_sums, scaled_from_rtl)

    for got, expected in zip(scaled_from_rtl, fixture["reference"]["ggml_scaled_float"]):
        assert abs(got - expected) <= FLOAT_TOLERANCE


def fp16_hex_to_float(value: str) -> float:
    raw = int(value, 16)
    return struct.unpack("<e", raw.to_bytes(2, byteorder="little"))[0]


def accumulate_scaled_block(
    fixture: dict,
    block_key: tuple[int, int],
    block_sums: list[int],
    scaled_from_rtl: list[float],
):
    group, q8_block = block_key
    if "groups" in fixture["selection"]:
        group_start = fixture["selection"]["groups"][0]
    else:
        group_start = fixture["selection"]["group"]
    group_index = group - group_start
    q1_scales = fixture["quantization"]["q1_scales_fp16_hex_by_group"][group_index]
    q8_scale = fp16_hex_to_float(
        fixture["quantization"]["q8_scales_fp16_hex_by_group"][group_index][q8_block]
    )

    for row in range(ROWS):
        q1_scale = fp16_hex_to_float(q1_scales[row])
        scaled_from_rtl[row] += q1_scale * q8_scale * block_sums[row]


@cocotb.test()
async def chip_w1a8_tile_transaction(dut):
    await init(dut)

    weights = [
        [1, 0, 1, 0],
        [0, 1, 1, 1],
    ]
    acts = [7, -8, -128, 5]
    seeds = [11, -13]
    expected = [
        dot_ref(weights[row], acts, seeds[row])
        for row in range(ROWS)
    ]

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
        expected = [
            dot_ref(weights[row], acts, seeds[row])
            for row in range(ROWS)
        ]
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
    await replay_fixture(
        dut,
        HERE / "fixtures" / "bonsai_blk0_attn_q_rows0_1_all_groups.json",
    )


@cocotb.test()
async def chip_matches_bonsai_q1_0_full_tensor_fixtures(dut):
    if os.getenv("BONSAI_FULL_TENSOR_TESTS") != "1":
        return

    fixture_dir = Path(os.environ["BONSAI_FULL_TENSOR_FIXTURE_DIR"])
    fixture_paths = sorted(fixture_dir.glob("*.json"))
    assert fixture_paths

    await init(dut)
    for fixture_path in fixture_paths:
        await replay_fixture(dut, fixture_path)
