import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ACT_WIDTH = 8
PSUM_WIDTH = 24
COLS = 4


def to_bits(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    sign = 1 << (width - 1)
    return raw - (1 << width) if raw & sign else raw


def pack_acts(lanes: list[int]) -> int:
    packed = 0
    for col, value in enumerate(lanes):
        packed |= to_bits(value, ACT_WIDTH) << (col * ACT_WIDTH)
    return packed


def unpack_acts(raw: int) -> list[int]:
    return [
        to_signed(raw >> (col * ACT_WIDTH), ACT_WIDTH)
        for col in range(COLS)
    ]


def read_psum(dut) -> int:
    return to_signed(int(dut.psum_out.value), PSUM_WIDTH)


def dot_ref(weights: list[int], acts: list[int], seed: int = 0) -> int:
    total = seed
    for weight, act in zip(weights, acts):
        total += act if weight else -act
    return total


async def init(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.clear.value = 0
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.act_in.value = 0
    dut.psum_in.value = 0
    dut.valid_in.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def cycle(dut, lanes=None, seed=0, valid=0, weight_load=0, weight_in=0):
    if lanes is None:
        lanes = [0] * COLS

    await FallingEdge(dut.clk)
    dut.clear.value = 0
    dut.weight_load.value = weight_load
    dut.weight_in.value = weight_in
    dut.act_in.value = pack_acts(lanes)
    dut.psum_in.value = to_bits(seed, PSUM_WIDTH)
    dut.valid_in.value = valid
    await RisingEdge(dut.clk)
    await ReadOnly()


async def load_weights(dut, weights: list[int]):
    # Serial shift chain enters at column 0. Load last column first so the
    # final stored PE weights are in normal column order.
    for bit in reversed(weights):
        await cycle(dut, weight_load=1, weight_in=bit)
    await cycle(dut)


async def clear_row(dut):
    await FallingEdge(dut.clk)
    dut.clear.value = 1
    dut.weight_load.value = 0
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.clear.value = 0


async def drive_one_dot(dut, acts: list[int], seed: int = 0):
    for step in range(COLS):
        lanes = [0] * COLS
        lanes[step] = acts[step]
        await cycle(dut, lanes=lanes, seed=seed if step == 0 else 0, valid=1 if step == 0 else 0)


@cocotb.test()
async def loads_weights_through_serial_chain(dut):
    await init(dut)

    weights = [1, 0, 1, 1]
    await load_weights(dut, weights)

    # weight_out is the stored weight from the last column.
    assert int(dut.weight_out.value) == weights[-1]

    acts = [3, 4, -5, 6]
    await drive_one_dot(dut, acts)

    assert int(dut.valid_out.value) == 1
    assert read_psum(dut) == dot_ref(weights, acts)


@cocotb.test()
async def computes_seeded_systolic_dot_product(dut):
    await init(dut)

    weights = [1, 0, 1, 0]
    acts = [7, -8, -128, 5]
    seed = -11

    await load_weights(dut, weights)
    await drive_one_dot(dut, acts, seed=seed)

    assert int(dut.valid_out.value) == 1
    assert read_psum(dut) == dot_ref(weights, acts, seed)


@cocotb.test()
async def supports_back_to_back_wavefronts(dut):
    await init(dut)

    weights = [0, 1, 1, 0]
    acts_by_op = [
        [10, 20, -30, -40],
        [-3, 4, 5, -6],
    ]
    seeds = [100, -7]
    expected = [
        dot_ref(weights, acts_by_op[op], seeds[op])
        for op in range(len(acts_by_op))
    ]

    await load_weights(dut, weights)

    got = []
    for step in range(COLS + len(acts_by_op) - 1):
        lanes = [0] * COLS
        for col in range(COLS):
            op = step - col
            if 0 <= op < len(acts_by_op):
                lanes[col] = acts_by_op[op][col]

        valid = 1 if step < len(acts_by_op) else 0
        seed = seeds[step] if step < len(seeds) else 0
        await cycle(dut, lanes=lanes, seed=seed, valid=valid)

        if int(dut.valid_out.value):
            got.append(read_psum(dut))

    assert got == expected


@cocotb.test()
async def forwards_activation_lanes_to_next_row(dut):
    await init(dut)
    await load_weights(dut, [1, 1, 1, 1])

    lanes = [1, -2, 3, -4]
    await cycle(dut, lanes=lanes, valid=0)

    assert unpack_acts(int(dut.act_out.value)) == lanes


@cocotb.test()
async def clear_flushes_pipeline_but_not_weights(dut):
    await init(dut)

    weights = [1, 0, 0, 1]
    await load_weights(dut, weights)
    await drive_one_dot(dut, [8, 7, 6, 5])
    assert int(dut.valid_out.value) == 1

    await clear_row(dut)
    assert int(dut.valid_out.value) == 0
    assert read_psum(dut) == 0
    assert int(dut.weight_out.value) == weights[-1]

    acts = [1, 2, 3, 4]
    await drive_one_dot(dut, acts)
    assert read_psum(dut) == dot_ref(weights, acts)
