import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cocotb

from common import (
    dot_ref,
    drive_and_sample,
    make_acts,
    make_seeds,
    make_weights,
    pack_lanes,
    start_clock_and_reset,
    unpack_lanes,
)


def read_psums(dut, rows: int, pw: int) -> list[int]:
    return unpack_lanes(int(dut.psum_out.value), rows, pw)


async def init(dut):
    dut.clear.value = 0
    dut.weight_load.value = 0
    dut.weight_in.value = 0
    dut.act_in.value = 0
    dut.psum_in.value = 0
    dut.valid_in.value = 0
    await start_clock_and_reset(dut)


async def cycle(dut, lanes=None, seeds=None, valids=None,
                weight_load: int = 0, weight_in=None):
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    aw = int(dut.ACT_WIDTH.value)
    pw = int(dut.PSUM_WIDTH.value)

    if lanes is None:
        lanes = [0] * cols
    if seeds is None:
        seeds = [0] * rows
    if valids is None:
        valids = [0] * rows
    if weight_in is None:
        weight_in = [0] * rows

    valid_bits = sum((1 if v else 0) << r for r, v in enumerate(valids))
    weight_bits = sum((1 if b else 0) << r for r, b in enumerate(weight_in))

    await drive_and_sample(
        dut,
        clear=0,
        weight_load=weight_load,
        weight_in=weight_bits,
        act_in=pack_lanes(lanes, aw),
        psum_in=pack_lanes(seeds, pw),
        valid_in=valid_bits,
    )


async def load_weights(dut, weights_by_row):
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    for col in reversed(range(cols)):
        await cycle(
            dut,
            weight_load=1,
            weight_in=[weights_by_row[r][col] for r in range(rows)],
        )
    await cycle(dut)


async def clear_array(dut):
    await drive_and_sample(dut, clear=1, weight_load=0, valid_in=0)


async def drive_one_vector_collect(dut, acts, seeds) -> list[int]:
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    pw = int(dut.PSUM_WIDTH.value)

    got = [None] * rows

    def sample():
        valid_mask = int(dut.valid_out.value)
        if not valid_mask:
            return
        values = read_psums(dut, rows, pw)
        for r in range(rows):
            if valid_mask & (1 << r):
                got[r] = values[r]

    # Column skew at the top edge; row skew is carried by valid/seed inputs
    # so each row starts when its activation reaches it.
    for step in range(cols + rows - 1):
        lanes = [0] * cols
        for c in range(cols):
            if step == c:
                lanes[c] = acts[c]
        valids    = [1 if step == r else 0 for r in range(rows)]
        row_seeds = [seeds[r] if step == r else 0 for r in range(rows)]
        await cycle(dut, lanes=lanes, seeds=row_seeds, valids=valids)
        sample()

    # Final drain: lower rows finish after the last top-edge activation injection.
    for _ in range(rows):
        if all(v is not None for v in got):
            break
        await cycle(dut)
        sample()

    missing = [r for r, v in enumerate(got) if v is None]
    assert not missing, f"no valid_out for rows={missing}, partial={got}"
    return got


@cocotb.test()
async def loads_one_weight_row_per_array_row(dut):
    await init(dut)
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)

    weights = make_weights(rows, cols, kind=0)
    await load_weights(dut, weights)

    acts = make_acts(cols, kind=0)
    got = await drive_one_vector_collect(dut, acts, [0] * rows)
    expected = [dot_ref(weights[r], acts) for r in range(rows)]
    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def computes_two_output_rows_for_one_activation_vector(dut):
    await init(dut)
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)

    weights = make_weights(rows, cols, kind=1)
    acts    = make_acts(cols, kind=1)
    seeds   = make_seeds(rows, kind=1)
    expected = [dot_ref(weights[r], acts, seeds[r]) for r in range(rows)]

    await load_weights(dut, weights)
    got = await drive_one_vector_collect(dut, acts, seeds)
    assert got == expected, f"got={got} expected={expected}"


@cocotb.test()
async def supports_back_to_back_activation_vectors(dut):
    await init(dut)
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    pw = int(dut.PSUM_WIDTH.value)

    weights     = make_weights(rows, cols, kind=2)
    acts_by_op  = [make_acts(cols, kind=k + 3) for k in range(3)]
    seeds_by_op = [make_seeds(rows, kind=k + 3) for k in range(3)]
    expected_by_op = [
        [dot_ref(weights[r], acts_by_op[op], seeds_by_op[op][r]) for r in range(rows)]
        for op in range(len(acts_by_op))
    ]

    await load_weights(dut, weights)

    got_by_row = [[] for _ in range(rows)]
    for step in range(cols + rows + len(acts_by_op) - 2):
        lanes = [0] * cols
        for c in range(cols):
            op = step - c
            if 0 <= op < len(acts_by_op):
                lanes[c] = acts_by_op[op][c]

        valids = [0] * rows
        seeds  = [0] * rows
        for r in range(rows):
            op = step - r
            if 0 <= op < len(acts_by_op):
                valids[r] = 1
                seeds[r]  = seeds_by_op[op][r]

        await cycle(dut, lanes=lanes, seeds=seeds, valids=valids)

        valid_mask = int(dut.valid_out.value)
        if valid_mask:
            values = read_psums(dut, rows, pw)
            for r in range(rows):
                if valid_mask & (1 << r):
                    got_by_row[r].append(values[r])

    expected_by_row = [
        [expected_by_op[op][r] for op in range(len(acts_by_op))]
        for r in range(rows)
    ]
    assert got_by_row == expected_by_row, (
        f"got={got_by_row} expected={expected_by_row}"
    )


@cocotb.test()
async def forwards_activations_out_of_bottom_row(dut):
    await init(dut)
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    aw = int(dut.ACT_WIDTH.value)

    await load_weights(dut, [[1] * cols for _ in range(rows)])

    lanes = make_acts(cols, kind=7)
    await cycle(dut, lanes=lanes)
    assert unpack_lanes(int(dut.act_out.value), cols, aw) == [0] * cols

    # Activations need ROWS-1 additional cycles to propagate through every row.
    for _ in range(rows - 1):
        await cycle(dut)
    assert unpack_lanes(int(dut.act_out.value), cols, aw) == lanes


@cocotb.test()
async def clear_flushes_array_but_keeps_weights(dut):
    await init(dut)
    rows = int(dut.ROWS.value)
    cols = int(dut.COLS.value)
    pw = int(dut.PSUM_WIDTH.value)

    weights = make_weights(rows, cols, kind=8)
    acts1 = make_acts(cols, kind=8)
    acts2 = make_acts(cols, kind=9)
    seeds = make_seeds(rows, kind=9)

    await load_weights(dut, weights)
    got = await drive_one_vector_collect(dut, acts1, [0] * rows)
    assert got == [dot_ref(weights[r], acts1) for r in range(rows)]

    await clear_array(dut)
    assert int(dut.valid_out.value) == 0
    assert read_psums(dut, rows, pw) == [0] * rows

    got = await drive_one_vector_collect(dut, acts2, seeds)
    expected = [dot_ref(weights[r], acts2, seeds[r]) for r in range(rows)]
    assert got == expected, f"got={got} expected={expected}"
