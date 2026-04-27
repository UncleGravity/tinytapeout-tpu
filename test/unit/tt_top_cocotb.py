import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ROWS = 2
COLS = 4

CMD_NOP = 0
CMD_CLEAR = 1
CMD_LOAD_WEIGHT = 2
CMD_LOAD_ACT = 3
CMD_LOAD_SEED = 4
CMD_START = 5
CMD_READ_RESULT = 6
CMD_STATUS = 7

STATUS_BUSY = 1 << 0
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
    dut.rst_n.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = 0
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


async def clear(dut):
    await command(dut, CMD_CLEAR)
    await nop(dut)


async def load_weights(dut, weights_by_row: list[list[int]]):
    for col in reversed(range(COLS)):
        bits = 0
        for row in range(ROWS):
            bits |= weights_by_row[row][col] << row
        await command(dut, CMD_LOAD_WEIGHT, data=bits)

    await nop(dut)
    status = await read_status(dut)
    assert status & STATUS_WEIGHT_DONE


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


@cocotb.test()
async def tt_wrapper_loads_computes_and_reads_results(dut):
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
        expected = [
            dot_ref(weights[row], acts, seeds[row])
            for row in range(ROWS)
        ]
        got = await run_vector(dut, acts, seeds)
        assert got == expected


@cocotb.test()
async def tt_wrapper_clear_resets_status_and_input_registers(dut):
    await init(dut)

    await load_weights(dut, [[1, 1, 1, 1], [1, 1, 1, 1]])
    await clear(dut)

    status = await read_status(dut)
    assert not (status & STATUS_DONE)
    assert not (status & STATUS_WEIGHT_DONE)

    got = await run_vector(dut, [1, 2, 3, 4], [0, 0])
    assert got == [10, 10]
