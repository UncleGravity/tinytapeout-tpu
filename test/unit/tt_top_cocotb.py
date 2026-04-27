import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ROWS = 2
COLS = 4
PSUM_WIDTH = 16
PSUM_BYTES = PSUM_WIDTH // 8

CMD_STATUS = 0
CMD_CLEAR = 1
CMD_SET_ADDR = 2
CMD_WRITE = 3
CMD_READ = 4
CMD_START = 5
CMD_NOP = 7

ADDR_ROW = 0
ADDR_COL = 1
ADDR_BYTE = 2
ADDR_BANK = 3

BANK_WEIGHT = 1
BANK_ACT = 2
BANK_SEED = 3
BANK_RESULT = 4

STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_WEIGHT_DONE = 1 << 2
STATUS_ALL_VALID = 1 << 3
STATUS_START_READY = 1 << 4
STATUS_ERROR = 1 << 6


def pack_ui(cmd: int, arg: int = 0) -> int:
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


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
    dut.rst_n.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = 0
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


async def set_addr(dut, addr_id: int, value: int):
    await command(dut, CMD_SET_ADDR, data=value, arg=addr_id)


async def set_bank(dut, bank: int):
    await set_addr(dut, ADDR_BANK, bank)


async def write_selected(dut, value: int):
    await command(dut, CMD_WRITE, data=value)


async def read_selected(dut) -> int:
    await command(dut, CMD_READ)
    return int(dut.uo_out.value) & 0xFF


async def clear(dut):
    await command(dut, CMD_CLEAR)
    await nop(dut)


async def load_weights(dut, weights_by_row: list[list[int]]):
    await set_bank(dut, BANK_WEIGHT)
    for row, weights in enumerate(weights_by_row):
        packed = 0
        for col, bit in enumerate(weights):
            packed |= (bit & 1) << col
        await set_addr(dut, ADDR_ROW, row)
        await set_addr(dut, ADDR_COL, 0)
        await write_selected(dut, packed)


async def load_acts(dut, acts: list[int]):
    await set_bank(dut, BANK_ACT)
    for col, act in enumerate(acts):
        await set_addr(dut, ADDR_COL, col)
        await write_selected(dut, to_bits(act, 8))


async def load_seed(dut, row: int, seed: int):
    raw = to_bits(seed, PSUM_WIDTH)
    await set_bank(dut, BANK_SEED)
    await set_addr(dut, ADDR_ROW, row)
    for byte in range(PSUM_BYTES):
        await set_addr(dut, ADDR_BYTE, byte)
        await write_selected(dut, (raw >> (8 * byte)) & 0xFF)


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
    await set_bank(dut, BANK_RESULT)
    await set_addr(dut, ADDR_ROW, row)
    for byte in range(PSUM_BYTES):
        await set_addr(dut, ADDR_BYTE, byte)
        raw |= (await read_selected(dut)) << (8 * byte)
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
