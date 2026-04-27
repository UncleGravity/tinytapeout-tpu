import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ACT_WIDTH = 8
PSUM_WIDTH = 16
ROWS = 2
COLS = 4


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


def pack_acts(values: list[int]) -> int:
    packed = 0
    for col, value in enumerate(values):
        packed |= to_bits(value, ACT_WIDTH) << (col * ACT_WIDTH)
    return packed


def pack_psums(values: list[int]) -> int:
    packed = 0
    for row, value in enumerate(values):
        packed |= to_bits(value, PSUM_WIDTH) << (row * PSUM_WIDTH)
    return packed


def unpack_psums(raw: int) -> list[int]:
    return [
        to_signed(raw >> (row * PSUM_WIDTH), PSUM_WIDTH)
        for row in range(ROWS)
    ]


def dot_ref(weights: list[int], acts: list[int], seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.weight_load_valid.value = 0
    dut.weight_load_bits.value = 0
    dut.start.value = 0
    dut.act_vector.value = 0
    dut.seed_in.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def idle_cycle(dut):
    await FallingEdge(dut.clk)
    dut.weight_load_valid.value = 0
    dut.weight_load_bits.value = 0
    dut.start.value = 0
    dut.act_vector.value = 0
    dut.seed_in.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()


async def load_weights(dut, weights_by_row: list[list[int]]):
    assert int(dut.weight_load_ready.value) == 1

    for col in reversed(range(COLS)):
        bits = 0
        for row in range(ROWS):
            bits |= weights_by_row[row][col] << row

        await FallingEdge(dut.clk)
        dut.weight_load_valid.value = 1
        dut.weight_load_bits.value = bits
        dut.start.value = 0
        await RisingEdge(dut.clk)
        await ReadOnly()

    assert int(dut.weight_load_done.value) == 1
    await idle_cycle(dut)
    assert int(dut.weight_load_done.value) == 0


async def start_compute(dut, acts: list[int], seeds: list[int]):
    assert int(dut.start_ready.value) == 1

    await FallingEdge(dut.clk)
    dut.weight_load_valid.value = 0
    dut.start.value = 1
    dut.act_vector.value = pack_acts(acts)
    dut.seed_in.value = pack_psums(seeds)
    await RisingEdge(dut.clk)
    await ReadOnly()

    assert int(dut.busy.value) == 1
    await FallingEdge(dut.clk)
    dut.start.value = 0


async def wait_done(dut, limit: int = 64) -> list[int]:
    for _ in range(limit):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.done.value):
            assert int(dut.result_valid.value) == (1 << ROWS) - 1
            return unpack_psums(int(dut.result_out.value))

    raise AssertionError("controller did not finish")


async def pulse_clear(dut):
    await FallingEdge(dut.clk)
    dut.clear.value = 1
    dut.weight_load_valid.value = 0
    dut.start.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()
    await FallingEdge(dut.clk)
    dut.clear.value = 0


@cocotb.test()
async def loads_weights_and_computes_aligned_results(dut):
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
    await start_compute(dut, acts, seeds)
    got = await wait_done(dut)

    assert got == expected
    assert int(dut.busy.value) == 0


@cocotb.test()
async def reuses_loaded_weights_for_back_to_back_vectors(dut):
    await init(dut)

    weights = [
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ]
    vectors = [
        ([10, 20, -30, -40], [100, -7]),
        ([-3, 4, 5, -6], [5, 6]),
        ([1, -2, 3, -4], [-3, 9]),
    ]

    await load_weights(dut, weights)

    for acts, seeds in vectors:
        expected = [
            dot_ref(weights[row], acts, seeds[row])
            for row in range(ROWS)
        ]
        await start_compute(dut, acts, seeds)
        got = await wait_done(dut)
        assert got == expected


@cocotb.test()
async def start_is_blocked_while_loading_weights(dut):
    await init(dut)

    await FallingEdge(dut.clk)
    dut.weight_load_valid.value = 1
    dut.weight_load_bits.value = 0b11
    dut.start.value = 1
    dut.act_vector.value = pack_acts([1, 2, 3, 4])
    dut.seed_in.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()

    assert int(dut.weight_load_ready.value) == 1
    assert int(dut.start_ready.value) == 0
    assert int(dut.busy.value) == 0


@cocotb.test()
async def clear_aborts_compute_but_keeps_weights(dut):
    await init(dut)

    weights = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
    ]
    acts = [8, 7, 6, 5]

    await load_weights(dut, weights)
    await start_compute(dut, acts, [0, 0])
    await ClockCycles(dut.clk, 2)
    await pulse_clear(dut)

    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert int(dut.result_valid.value) == 0

    seeds = [3, -4]
    expected = [
        dot_ref(weights[row], acts, seeds[row])
        for row in range(ROWS)
    ]
    await start_compute(dut, acts, seeds)
    got = await wait_done(dut)
    assert got == expected
