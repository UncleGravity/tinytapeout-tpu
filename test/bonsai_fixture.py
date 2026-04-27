import json
import struct
from itertools import groupby
from pathlib import Path

from tt_protocol import COLS, ROWS, load_weights, run_vector


Q1_GROUP_SIZE = 128
Q8_BLOCK_SIZE = 32
FLOAT_TOLERANCE = 1e-4


def load_fixture(path: Path) -> dict:
    fixture = json.loads(path.read_text())
    validate_fixture_file(fixture)
    return fixture


def validate_fixture_file(fixture: dict):
    if fixture.get("kind") == "tensor":
        assert fixture["schema_version"] == 1
        assert fixture["tile"] == {"rows": ROWS, "cols": COLS}
        assert fixture["row_tiles"]
        for row_tile in fixture["row_tiles"]:
            validate_fixture(row_tile)
        return

    validate_fixture(fixture)


def validate_fixture(fixture: dict):
    assert fixture["schema_version"] == 1
    assert fixture["source"]["type"] == "q1_0"
    assert fixture["tile"] == {"rows": ROWS, "cols": COLS}
    assert fixture["quantization"]["q1_0_group_size"] == Q1_GROUP_SIZE
    assert fixture["quantization"]["q8_0_block_size"] == Q8_BLOCK_SIZE
    assert Q8_BLOCK_SIZE % COLS == 0
    assert len(fixture["reference"]["integer_final"]) == ROWS
    assert len(fixture["reference"]["ggml_scaled_float"]) == ROWS

    q1_scales = fixture["quantization"]["q1_scales_fp16_hex_by_group"]
    q8_scales = fixture["quantization"]["q8_scales_fp16_hex_by_group"]
    assert q1_scales
    assert len(q1_scales) == len(q8_scales)

    for txn in fixture["transactions"]:
        assert len(txn["weights"]) == ROWS
        assert all(len(row_weights) == COLS for row_weights in txn["weights"])
        assert len(txn["acts"]) == COLS
        assert len(txn["seeds"]) == ROWS
        assert len(txn["expected"]) == ROWS
        assert txn["cols"][0] % COLS == 0


async def replay_fixture(dut, path: Path):
    fixture = load_fixture(path)
    if fixture.get("kind") == "tensor":
        for row_tile in fixture["row_tiles"]:
            await replay_loaded_fixture(dut, row_tile)
        return

    await replay_loaded_fixture(dut, fixture)


async def replay_loaded_fixture(dut, fixture: dict):
    scaled_from_rtl = [0.0] * ROWS

    for block_key, txns_iter in groupby(fixture["transactions"], key=q8_block_key):
        block_sum = await replay_q8_block(dut, list(txns_iter))
        add_scaled_block(fixture, block_key, block_sum, scaled_from_rtl)

    for got, expected in zip(scaled_from_rtl, fixture["reference"]["ggml_scaled_float"]):
        assert abs(got - expected) <= FLOAT_TOLERANCE


def q8_block_key(txn: dict) -> tuple[int, int]:
    return txn["group"], (txn["cols"][0] % Q1_GROUP_SIZE) // Q8_BLOCK_SIZE


async def replay_q8_block(dut, txns: list[dict]) -> list[int]:
    psum = [0] * ROWS

    for txn in txns:
        delta = [
            txn["expected"][row] - txn["seeds"][row]
            for row in range(ROWS)
        ]
        expected = [
            psum[row] + delta[row]
            for row in range(ROWS)
        ]

        await load_weights(dut, txn["weights"])
        psum = await run_vector(dut, txn["acts"], psum)
        assert psum == expected

    return psum


def add_scaled_block(
    fixture: dict,
    block_key: tuple[int, int],
    block_sum: list[int],
    scaled: list[float],
):
    group, q8_block = block_key
    group_index = group - group_start(fixture)
    q1_scales = fixture["quantization"]["q1_scales_fp16_hex_by_group"][group_index]
    q8_scale = fp16_hex_to_float(
        fixture["quantization"]["q8_scales_fp16_hex_by_group"][group_index][q8_block]
    )

    for row in range(ROWS):
        scaled[row] += fp16_hex_to_float(q1_scales[row]) * q8_scale * block_sum[row]


def group_start(fixture: dict) -> int:
    selection = fixture["selection"]
    return selection["groups"][0] if "groups" in selection else selection["group"]


def fp16_hex_to_float(value: str) -> float:
    raw = int(value, 16)
    return struct.unpack("<e", raw.to_bytes(2, byteorder="little"))[0]
