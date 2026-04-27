import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


ACT_WIDTH = 8
PSUM_WIDTH = 24
ROWS = 2
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
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.act_in.value = 0
    dut.psum_in.value = 0
    dut.valid_in.value = 0
    await ClockCycles(dut.clk, 4)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def cycle(dut, lanes=None, seeds=None, valids=None, weight_load=0, weight_in=None):
    if lanes is None:
        lanes = [0] * COLS
    if seeds is None:
        seeds = [0] * ROWS
    if valids is None:
        valids = [0] * ROWS
    if weight_in is None:
        weight_in = [0] * ROWS

    valid_bits = 0
    for row, valid in enumerate(valids):
        valid_bits |= (1 if valid else 0) << row

    weight_bits = 0
    for row, bit in enumerate(weight_in):
        weight_bits |= (1 if bit else 0) << row

    await FallingEdge(dut.clk)
    dut.clear.value = 0
    dut.weight_load.value = weight_load
    dut.weight_in.value = weight_bits
    dut.act_in.value = pack_acts(lanes)
    dut.psum_in.value = pack_psums(seeds)
    dut.valid_in.value = valid_bits
    await RisingEdge(dut.clk)
    await ReadOnly()


async def load_weights(dut, weights_by_row: list[list[int]]):
    for col in reversed(range(COLS)):
        await cycle(
            dut,
            weight_load=1,
            weight_in=[weights_by_row[row][col] for row in range(ROWS)],
        )
    await cycle(dut)


async def clear_array(dut):
    await FallingEdge(dut.clk)
    dut.clear.value = 1
    dut.weight_load.value = 0
    dut.valid_in.value = 0
    await RisingEdge(dut.clk)
    await ReadOnly()


async def drive_one_vector(dut, acts: list[int], seeds: list[int]):
    # Column skew happens at the top edge. Row skew is carried by valid/seed
    # inputs so each row starts when x0 reaches that row.
    for step in range(COLS + ROWS - 1):
        lanes = [0] * COLS
        for col in range(COLS):
            if step == col:
                lanes[col] = acts[col]

        valids = [1 if step == row else 0 for row in range(ROWS)]
        row_seeds = [
            seeds[row] if step == row else 0
            for row in range(ROWS)
        ]
        await cycle(dut, lanes=lanes, seeds=row_seeds, valids=valids)


async def drive_one_vector_collect(dut, acts: list[int], seeds: list[int]) -> list[int]:
    got = [None] * ROWS

    for step in range(COLS + ROWS - 1):
        lanes = [0] * COLS
        for col in range(COLS):
            if step == col:
                lanes[col] = acts[col]

        valids = [1 if step == row else 0 for row in range(ROWS)]
        row_seeds = [
            seeds[row] if step == row else 0
            for row in range(ROWS)
        ]
        await cycle(dut, lanes=lanes, seeds=row_seeds, valids=valids)

        valid_mask = int(dut.valid_out.value)
        values = unpack_psums(int(dut.psum_out.value))
        for row in range(ROWS):
            if valid_mask & (1 << row):
                got[row] = values[row]

    # The lowest rows can finish after the last top-edge activation injection.
    for _ in range(ROWS):
        if all(value is not None for value in got):
            break
        await cycle(dut)
        valid_mask = int(dut.valid_out.value)
        values = unpack_psums(int(dut.psum_out.value))
        for row in range(ROWS):
            if valid_mask & (1 << row):
                got[row] = values[row]

    assert all(value is not None for value in got), f"missing row outputs: {got}"
    return got


@cocotb.test()
async def loads_one_weight_row_per_array_row(dut):
    await init(dut)

    weights = [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
    ]
    await load_weights(dut, weights)

    # Each row has its own serial load lane; weight_out is that row's last PE.
    got = int(dut.weight_out.value)
    assert got == ((weights[0][-1] << 0) | (weights[1][-1] << 1))


@cocotb.test()
async def computes_two_output_rows_for_one_activation_vector(dut):
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
    got = await drive_one_vector_collect(dut, acts, seeds)

    assert got == expected


@cocotb.test()
async def supports_back_to_back_activation_vectors(dut):
    await init(dut)

    weights = [
        [0, 1, 1, 0],
        [1, 1, 0, 0],
    ]
    acts_by_op = [
        [10, 20, -30, -40],
        [-3, 4, 5, -6],
        [1, -2, 3, -4],
    ]
    seeds_by_op = [
        [100, -7],
        [5, 6],
        [-3, 9],
    ]
    expected_by_op = [
        [
            dot_ref(weights[row], acts_by_op[op], seeds_by_op[op][row])
            for row in range(ROWS)
        ]
        for op in range(len(acts_by_op))
    ]

    await load_weights(dut, weights)

    got_by_row = [[] for _ in range(ROWS)]
    for step in range(COLS + ROWS + len(acts_by_op) - 2):
        lanes = [0] * COLS
        for col in range(COLS):
            op = step - col
            if 0 <= op < len(acts_by_op):
                lanes[col] = acts_by_op[op][col]

        valids = [0] * ROWS
        seeds = [0] * ROWS
        for row in range(ROWS):
            op = step - row
            if 0 <= op < len(acts_by_op):
                valids[row] = 1
                seeds[row] = seeds_by_op[op][row]

        await cycle(dut, lanes=lanes, seeds=seeds, valids=valids)

        valid_mask = int(dut.valid_out.value)
        if valid_mask:
            values = unpack_psums(int(dut.psum_out.value))
            for row in range(ROWS):
                if valid_mask & (1 << row):
                    got_by_row[row].append(values[row])

    expected_by_row = [
        [expected_by_op[op][row] for op in range(len(acts_by_op))]
        for row in range(ROWS)
    ]
    assert got_by_row == expected_by_row


@cocotb.test()
async def forwards_activations_out_of_bottom_row(dut):
    await init(dut)
    await load_weights(dut, [[1] * COLS, [1] * COLS])

    lanes = [1, -2, 3, -4]
    await cycle(dut, lanes=lanes)
    assert unpack_acts(int(dut.act_out.value)) == [0] * COLS

    await cycle(dut)
    assert unpack_acts(int(dut.act_out.value)) == lanes


@cocotb.test()
async def clear_flushes_array_but_keeps_weights(dut):
    await init(dut)

    weights = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
    ]
    acts = [8, 7, 6, 5]

    await load_weights(dut, weights)
    got = await drive_one_vector_collect(dut, acts, [0, 0])
    assert got == [
        dot_ref(weights[row], acts, 0)
        for row in range(ROWS)
    ]

    await clear_array(dut)
    assert int(dut.valid_out.value) == 0
    assert unpack_psums(int(dut.psum_out.value)) == [0, 0]

    got = await drive_one_vector_collect(dut, acts, [3, -4])
    expected = [
        dot_ref(weights[row], acts, [3, -4][row])
        for row in range(ROWS)
    ]
    assert got == expected
