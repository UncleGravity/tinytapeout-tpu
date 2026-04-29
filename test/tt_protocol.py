"""Chip-level protocol helpers for tt_um_unclegravity_tpu.

Tile shape (ROWS / COLS / ACT_WIDTH / PSUM_WIDTH) is parsed from project.v's
localparams, so changing them is enough to retarget the tests. The Tile is stashed
on dut as `dut._tile` so subsequent helpers can reach it without the caller
threading it through every call.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from common import (
    BITS_PER_BYTE,
    drive_and_sample,
    start_clock_and_reset,
    to_bits,
    to_signed,
)


_PROJECT_V = Path(__file__).parent.parent / "src" / "rtl" / "project.v"


# Command opcodes (must match project.v).
CMD_STATUS = 0
CMD_CLEAR  = 1
CMD_LDW    = 2
CMD_LDA    = 3
CMD_SEED   = 4
CMD_START  = 5
CMD_RDP    = 6
CMD_NOP    = 7

# Status byte bits (must match project.v).
STATUS_BUSY        = 1 << 0
STATUS_DONE        = 1 << 1
STATUS_WEIGHT_DONE = 1 << 2
STATUS_ALL_VALID   = 1 << 3
STATUS_START_READY = 1 << 4
STATUS_ERROR       = 1 << 6

_STATUS_FLAG_NAMES = {
    STATUS_BUSY:        "BUSY",
    STATUS_DONE:        "DONE",
    STATUS_WEIGHT_DONE: "WEIGHT_DONE",
    STATUS_ALL_VALID:   "ALL_VALID",
    STATUS_START_READY: "START_READY",
    STATUS_ERROR:       "ERROR",
}


@dataclass(frozen=True)
class Tile:
    rows: int
    cols: int
    act_width: int
    psum_width: int

    @property
    def psum_bytes(self) -> int:
        return (self.psum_width + BITS_PER_BYTE - 1) // BITS_PER_BYTE

    @property
    def row_bits(self) -> int:
        return max(1, (self.rows - 1).bit_length())

    @property
    def col_bits(self) -> int:
        return max(1, (self.cols - 1).bit_length())

    @property
    def byte_bits(self) -> int:
        return max(1, (self.psum_bytes - 1).bit_length())


def tile(dut) -> Tile:
    """The Tile that init() attached to this dut."""
    return dut._tile


def _read_tile() -> Tile:
    text = _PROJECT_V.read_text()
    def lp(name: str) -> int:
        m = re.search(rf"localparam\s+{name}\s*=\s*(\d+)", text)
        if not m:
            raise RuntimeError(f"localparam {name} not found in {_PROJECT_V}")
        return int(m.group(1))
    return Tile(
        rows       = lp("ROWS"),
        cols       = lp("COLS"),
        act_width  = lp("ACT_WIDTH"),
        psum_width = lp("PSUM_WIDTH"),
    )


def pack_ui(cmd: int, arg: int = 0) -> int:
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


def encode_row_byte_arg(dut, row: int, byte: int) -> int:
    t = tile(dut)
    return (row & ((1 << t.row_bits) - 1)) | (byte << t.row_bits)


def encode_col_arg(dut, col: int) -> int:
    return col & ((1 << tile(dut).col_bits) - 1)


def encode_row_arg(dut, row: int) -> int:
    return row & ((1 << tile(dut).row_bits) - 1)


async def init(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await start_clock_and_reset(dut)
    dut._tile = _read_tile()


async def command(dut, cmd: int, data: int = 0, arg: int = 0):
    await drive_and_sample(
        dut,
        ui_in=pack_ui(cmd, arg=arg),
        uio_in=data & 0xFF,
    )


async def nop(dut):
    await command(dut, CMD_NOP)


async def read_status(dut) -> int:
    await command(dut, CMD_STATUS)
    return int(dut.uo_out.value) & 0xFF


def _flag_names(mask: int) -> str:
    names = [name for bit, name in _STATUS_FLAG_NAMES.items() if mask & bit]
    return "+".join(names) if names else "(none)"


async def expect_status(dut, *, must_have: int = 0, must_not_have: int = 0) -> int:
    status = await read_status(dut)
    missing   = must_have & ~status
    forbidden = must_not_have & status
    if missing or forbidden:
        raise AssertionError(
            f"status=0x{status:02x}: missing={_flag_names(missing)} "
            f"forbidden={_flag_names(forbidden)}"
        )
    return status


async def clear(dut):
    await command(dut, CMD_CLEAR)
    await nop(dut)


async def load_weights(dut, weights_by_row):
    t = tile(dut)
    assert t.cols <= BITS_PER_BYTE, (
        f"LDW packs all bits in one byte; COLS={t.cols} would need a chunk arg"
    )
    for row, weights in enumerate(weights_by_row):
        packed = 0
        for col, bit in enumerate(weights):
            packed |= (bit & 1) << col
        await command(dut, CMD_LDW, data=packed, arg=encode_row_arg(dut, row))


async def load_acts(dut, acts):
    for col, act in enumerate(acts):
        await command(
            dut,
            CMD_LDA,
            data=to_bits(act, BITS_PER_BYTE),
            arg=encode_col_arg(dut, col),
        )


async def load_seed(dut, row: int, seed: int):
    t = tile(dut)
    raw = to_bits(seed, t.psum_width)
    for byte in range(t.psum_bytes):
        await command(
            dut,
            CMD_SEED,
            data=(raw >> (BITS_PER_BYTE * byte)) & 0xFF,
            arg=encode_row_byte_arg(dut, row, byte),
        )


async def start(dut):
    await expect_status(dut, must_have=STATUS_START_READY)
    await command(dut, CMD_START)
    await nop(dut)


async def wait_done(dut, limit: int = 128) -> int:
    for _ in range(limit):
        status = await expect_status(dut, must_not_have=STATUS_ERROR)
        if status & STATUS_DONE:
            return status
    raise AssertionError(
        f"top wrapper did not report DONE within {limit} status reads"
    )


async def read_result(dut, row: int) -> int:
    t = tile(dut)
    raw = 0
    for byte in range(t.psum_bytes):
        await command(dut, CMD_RDP, arg=encode_row_byte_arg(dut, row, byte))
        raw |= (int(dut.uo_out.value) & 0xFF) << (BITS_PER_BYTE * byte)
    return to_signed(raw, t.psum_width)


async def run_vector(dut, acts, seeds) -> list[int]:
    t = tile(dut)
    await load_acts(dut, acts)
    for row, seed in enumerate(seeds):
        await load_seed(dut, row, seed)
    await start(dut)
    status = await wait_done(dut)
    required = STATUS_ALL_VALID | STATUS_WEIGHT_DONE
    if (status & required) != required:
        raise AssertionError(
            f"after wait_done status=0x{status:02x}: "
            f"missing={_flag_names(required & ~status)}"
        )
    return [await read_result(dut, row) for row in range(t.rows)]
