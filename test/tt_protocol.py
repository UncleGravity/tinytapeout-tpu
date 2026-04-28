import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ROWS = 2
COLS = 4
PSUM_WIDTH = 16
PSUM_BYTES = PSUM_WIDTH // 8

CMD_STATUS = 0
CMD_CLEAR  = 1
CMD_LDW    = 2
CMD_LDA    = 3
CMD_SEED   = 4
CMD_START  = 5
CMD_RDP    = 6
CMD_NOP    = 7

STATUS_BUSY        = 1 << 0
STATUS_DONE        = 1 << 1
STATUS_WEIGHT_DONE = 1 << 2
STATUS_ALL_VALID   = 1 << 3
STATUS_START_READY = 1 << 4
STATUS_ERROR       = 1 << 6


def _row_bits() -> int:
    return max(1, (ROWS - 1).bit_length())


def _byte_bits() -> int:
    return max(1, (PSUM_BYTES - 1).bit_length())


def _col_bits() -> int:
    return max(1, (COLS - 1).bit_length())


def pack_ui(cmd: int, arg: int = 0) -> int:
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


def encode_row_byte_arg(row: int, byte: int) -> int:
    return (row & ((1 << _row_bits()) - 1)) | (byte << _row_bits())


def encode_col_arg(col: int) -> int:
    return col & ((1 << _col_bits()) - 1)


def encode_row_arg(row: int) -> int:
    return row & ((1 << _row_bits()) - 1)


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


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


async def command(dut, cmd: int, data: int = 0, arg: int = 0):
    await FallingEdge(dut.clk)
    dut.ui_in.value = pack_ui(cmd, arg=arg)
    dut.uio_in.value = data & 0xFF
    await RisingEdge(dut.clk)
    await ReadOnly()


async def nop(dut):
    await command(dut, CMD_NOP)


async def read_status(dut) -> int:
    await command(dut, CMD_STATUS)
    return int(dut.uo_out.value) & 0xFF


async def clear(dut):
    await command(dut, CMD_CLEAR)
    await nop(dut)


async def load_weights(dut, weights_by_row: list[list[int]]):
    # One LDW per row packs all COLS bits (assumes COLS <= 8).
    assert COLS <= 8, "LDW packs all bits in one byte; COLS > 8 would need a chunk arg"
    for row, weights in enumerate(weights_by_row):
        packed = 0
        for col, bit in enumerate(weights):
            packed |= (bit & 1) << col
        await command(dut, CMD_LDW, data=packed, arg=encode_row_arg(row))


async def load_acts(dut, acts: list[int]):
    for col, act in enumerate(acts):
        await command(dut, CMD_LDA, data=to_bits(act, 8), arg=encode_col_arg(col))


async def load_seed(dut, row: int, seed: int):
    raw = to_bits(seed, PSUM_WIDTH)
    for byte in range(PSUM_BYTES):
        await command(
            dut,
            CMD_SEED,
            data=(raw >> (8 * byte)) & 0xFF,
            arg=encode_row_byte_arg(row, byte),
        )


async def start(dut):
    status = await read_status(dut)
    assert status & STATUS_START_READY
    await command(dut, CMD_START)
    await nop(dut)


async def wait_done(dut, limit: int = 128) -> int:
    for _ in range(limit):
        status = await read_status(dut)
        assert not (status & STATUS_ERROR)
        if status & STATUS_DONE:
            return status
    raise AssertionError("top wrapper did not report done")


async def read_result(dut, row: int) -> int:
    raw = 0
    for byte in range(PSUM_BYTES):
        await command(dut, CMD_RDP, arg=encode_row_byte_arg(row, byte))
        raw |= (int(dut.uo_out.value) & 0xFF) << (8 * byte)
    return to_signed(raw, PSUM_WIDTH)


async def run_vector(dut, acts: list[int], seeds: list[int]) -> list[int]:
    await load_acts(dut, acts)
    for row, seed in enumerate(seeds):
        await load_seed(dut, row, seed)
    await start(dut)
    status = await wait_done(dut)
    assert status & STATUS_ALL_VALID
    assert status & STATUS_WEIGHT_DONE
    return [await read_result(dut, row) for row in range(ROWS)]
