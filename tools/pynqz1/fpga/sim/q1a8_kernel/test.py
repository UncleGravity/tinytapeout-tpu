"""Dual-stream (v4) Q1A8 kernel cocotb tests.

The v4 kernel exposes two AXIS slave ports:

  * S_AXIS_ACTS — receives one column's acts + scales once at the start;
    the kernel buffers them in BRAM and reuses across rowblocks.
  * S_AXIS      — receives the per-rowblock weight stream (scales + wbits
    only — no act repetition).

Compared to the v3 single-stream kernel this halves DMA traffic for
typical matmuls (acts no longer get re-sent per rowblock) and removes
the host-side merge step entirely (the weights stream is byte-identical
to the on-DDR packed_weights layout).
"""

from __future__ import annotations

import random
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

CLK_PERIOD_NS = 10
ROWS = 8
Q1_BLOCK = 128
Q8_BLOCK = 32
Q8_SUBBLOCKS = Q1_BLOCK // Q8_BLOCK


def fp16_bits_to_fp32_bits(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return sign << 31
    return (sign << 31) | ((exp + 112) << 23) | (mant << 13)


def int_to_fp32_bits(value, width=14):
    if value == 0:
        return 0
    mag_w = width - 1
    sign = 1 if value < 0 else 0
    mag = -value if value < 0 else value
    msb_pos = mag.bit_length() - 1
    mag_norm = mag << (mag_w - 1 - msb_pos)
    mant_top = mag_norm & ((1 << (mag_w - 1)) - 1)
    mantissa = mant_top << (23 - (mag_w - 1))
    exponent = 127 + msb_pos
    return (sign << 31) | (exponent << 23) | mantissa


def fp32_mul_trunc(a, b):
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    if ea == 0 or eb == 0:
        return 0
    sr = sa ^ sb
    prod = ((1 << 23) | ma) * ((1 << 23) | mb)
    renorm = (prod >> 47) & 1
    if renorm:
        mr = (prod >> 24) & 0x7FFFFF
        er = ea + eb - 127 + 1
    else:
        mr = (prod >> 23) & 0x7FFFFF
        er = ea + eb - 127
    if er <= 0:
        return sr << 31
    if er >= 255:
        return (sr << 31) | (0xFE << 23) | 0x7FFFFF
    return (sr << 31) | (er << 23) | mr


def fp32_add_trunc(a, b):
    sa, ea, ma = (a >> 31) & 1, (a >> 23) & 0xFF, a & 0x7FFFFF
    sb, eb, mb = (b >> 31) & 1, (b >> 23) & 0xFF, b & 0x7FFFFF
    if ea == 0:
        return b
    if eb == 0:
        return a
    a_ge_b = ea >= eb
    sign_big, exp_big, mant_big = (sa, ea, (1 << 23) | ma) if a_ge_b else (sb, eb, (1 << 23) | mb)
    sign_small, _, mant_small = (sb, eb, (1 << 23) | mb) if a_ge_b else (sa, ea, (1 << 23) | ma)
    exp_diff = exp_big - (eb if a_ge_b else ea)
    mant_small_aligned = 0 if exp_diff > 24 else (mant_small >> exp_diff)
    if exp_diff == 0 and mant_small_aligned > mant_big:
        m1, m2, result_sign = mant_small_aligned, mant_big, sign_small
    else:
        m1, m2, result_sign = mant_big, mant_small_aligned, sign_big
    mant_sum = m1 + m2 if sign_big == sign_small else m1 - m2
    if mant_sum == 0:
        return 0
    lead_pos = mant_sum.bit_length() - 1
    if lead_pos > 23:
        right = lead_pos - 23
        mant_norm = mant_sum >> right
        exp_signed = exp_big + right
    elif lead_pos < 23:
        left = 23 - lead_pos
        mant_norm = mant_sum << left
        exp_signed = exp_big - left
    else:
        mant_norm = mant_sum
        exp_signed = exp_big
    if exp_signed <= 0:
        return result_sign << 31
    if exp_signed >= 255:
        return (result_sign << 31) | (0xFE << 23) | 0x7FFFFF
    return (result_sign << 31) | (exp_signed << 23) | (mant_norm & 0x7FFFFF)


def fp16_float_to_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def contribution(weight_bits, acts, ws_h, as_h):
    sub_sum = sum(acts[i] if (weight_bits >> i) & 1 else -acts[i] for i in range(Q8_BLOCK))
    ws_f32 = fp16_bits_to_fp32_bits(ws_h)
    as_f32 = fp16_bits_to_fp32_bits(as_h)
    ss_f32 = int_to_fp32_bits(sub_sum, width=14)
    return fp32_mul_trunc(fp32_mul_trunc(ws_f32, as_f32), ss_f32)


def golden_rowblock(weight_rows, acts, act_scales):
    """Run the matmul golden for one rowblock (ROWS lanes).

    weight_rows[row] = [(ws_h, bits128), ...] one entry per Q1 block.
    acts = full column acts (q1_blocks * Q1_BLOCK ints).
    act_scales = q1_blocks * Q8_SUBBLOCKS fp16 scale words.
    """
    out = []
    for row in range(ROWS):
        acc = 0
        for q1_index, (ws_h, bits128) in enumerate(weight_rows[row]):
            for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
                q8_index = q1_index * Q8_SUBBLOCKS + q8_local // Q8_BLOCK
                act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
                wb = (bits128 >> q8_local) & 0xFFFF_FFFF
                acc = fp32_add_trunc(
                    acc, contribution(wb, act_block, ws_h, act_scales[q8_index])
                )
        out.append(acc)
    return out


def random_rowblock(seed, q1_blocks, active_rows=ROWS):
    """Random weights for one rowblock + a random acts column.

    The acts/scales are shared by every rowblock in a real matmul; the
    test layer above this picks acts ONCE per matmul, not per rowblock.
    """
    rng = random.Random(seed)
    weight_rows = []
    for row in range(ROWS):
        blocks = []
        for q1_index in range(q1_blocks):
            if row < active_rows:
                ws_h = fp16_float_to_bits(0.01 + 0.003 * ((row + q1_index) % 17))
                bits128 = rng.getrandbits(Q1_BLOCK)
            else:
                ws_h = 0
                bits128 = 0
            blocks.append((ws_h, bits128))
        weight_rows.append(blocks)
    acts = [rng.randint(-128, 127) for _ in range(q1_blocks * Q1_BLOCK)]
    act_scales = [
        fp16_float_to_bits(0.001 + 0.0007 * ((i * 5 + 3) % 19))
        for i in range(q1_blocks * Q8_SUBBLOCKS)
    ]
    return weight_rows, acts, act_scales


# -- v4 stream packers ---------------------------------------------------


def pack_acts_stream(acts, act_scales, q1_blocks):
    """Pack one matmul column's acts/scales into the S_AXIS_ACTS beats.

    Per Q1 block × per sub-block: 4 beats × 8 int8 + 1 beat fp16 scale.
    """
    beats = []
    for q1_index in range(q1_blocks):
        for sub in range(Q8_SUBBLOCKS):
            q8_index = q1_index * Q8_SUBBLOCKS + sub
            act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
            for beat in range(4):
                word = 0
                for byte in range(8):
                    word |= (act_block[beat * 8 + byte] & 0xFF) << (byte * 8)
                beats.append(word)
            beats.append(act_scales[q8_index])
    return beats


def pack_weights_stream(weight_rows, q1_blocks):
    """Pack one rowblock's worth of weight stream beats (scales + wbits only).

    Matches the on-DDR packed_weights layout the host stores at upload.
    """
    beats = []
    scale_beats = (ROWS + 3) // 4
    wbits_beats = (ROWS + 1) // 2

    for q1_index in range(q1_blocks):
        for beat in range(scale_beats):
            word = 0
            for local in range(4):
                lane = beat * 4 + local
                scale = weight_rows[lane][q1_index][0] if lane < ROWS else 0
                word |= scale << (local * 16)
            beats.append(word)

        for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
            for beat in range(wbits_beats):
                word = 0
                for local in range(2):
                    lane = beat * 2 + local
                    bits = (
                        (weight_rows[lane][q1_index][1] >> q8_local) & 0xFFFF_FFFF
                        if lane < ROWS
                        else 0
                    )
                    word |= bits << (local * 32)
                beats.append(word)

    return beats


# -- bench helpers -------------------------------------------------------


async def clk_setup(dut):
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, unit="ns").start())


async def reset(dut):
    dut.rst_n.value = 0
    dut.start_kernel.value = 0
    dut.num_q1_blocks.value = 0
    dut.num_rowblocks.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_acts_tvalid.value = 0
    dut.s_axis_acts_tdata.value = 0
    dut.m_axis_tready.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def push_beat_to(dut, tdata_sig, tvalid_sig, tready_sig, data, stall_cycles=0):
    tvalid_sig.value = 0
    for _ in range(stall_cycles):
        await RisingEdge(dut.clk)
    tdata_sig.value = data
    tvalid_sig.value = 1
    while True:
        await ReadOnly()
        if int(tready_sig.value) == 1:
            break
        await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def stream_acts(dut, beats, stall_rng=None):
    for beat in beats:
        stall = stall_rng.randint(0, 2) if stall_rng else 0
        await push_beat_to(
            dut, dut.s_axis_acts_tdata, dut.s_axis_acts_tvalid,
            dut.s_axis_acts_tready, beat, stall_cycles=stall,
        )
    dut.s_axis_acts_tvalid.value = 0


async def stream_weights(dut, beats, stall_rng=None):
    for beat in beats:
        stall = stall_rng.randint(0, 2) if stall_rng else 0
        await push_beat_to(
            dut, dut.s_axis_tdata, dut.s_axis_tvalid,
            dut.s_axis_tready, beat, stall_cycles=stall,
        )
    dut.s_axis_tvalid.value = 0


async def capture_output(dut, num_beats, ready_rng=None):
    def pick_ready():
        if ready_rng is None:
            return 1
        return 1 if ready_rng.random() < 0.7 else 0

    data = []
    last = []
    dut.m_axis_tready.value = pick_ready()
    while len(data) < num_beats:
        await ReadOnly()
        tvalid_now = int(dut.m_axis_tvalid.value)
        tready_now = int(dut.m_axis_tready.value)
        if tready_now and tvalid_now == 1:
            data.append(int(dut.m_axis_tdata.value))
            last.append(int(dut.m_axis_tlast.value))
        await RisingEdge(dut.clk)
        dut.m_axis_tready.value = pick_ready()
    dut.m_axis_tready.value = 0
    return data, last


def unpack_axis_results(beats):
    out = []
    for beat in beats:
        out.append(beat & 0xFFFF_FFFF)
        out.append((beat >> 32) & 0xFFFF_FFFF)
    return out


async def run_matmul(
    dut,
    acts_beats,
    weights_beats,
    num_q1_blocks,
    num_rowblocks,
    acts_stall_rng=None,
    weights_stall_rng=None,
    output_ready_rng=None,
):
    """Drive a full matmul column: acts once, then weights rowblock by rowblock."""
    dut.num_q1_blocks.value = num_q1_blocks
    dut.num_rowblocks.value = num_rowblocks
    dut.start_kernel.value = 1
    await RisingEdge(dut.clk)
    dut.start_kernel.value = 0

    capture_task = cocotb.start_soon(
        capture_output(dut, num_rowblocks * 4, ready_rng=output_ready_rng)
    )
    # Both streams can be in flight in parallel; the kernel will gate s_axis
    # behind LOAD_ACTS until all acts are buffered.
    acts_task = cocotb.start_soon(stream_acts(dut, acts_beats, stall_rng=acts_stall_rng))
    weights_task = cocotb.start_soon(
        stream_weights(dut, weights_beats, stall_rng=weights_stall_rng)
    )
    await acts_task
    await weights_task
    out_beats, tlast = await capture_task

    for _ in range(64):
        await RisingEdge(dut.clk)
        if int(dut.kernel_done.value) == 1:
            break
    else:
        raise AssertionError("kernel_done never fired")

    return out_beats, tlast


def diff_msg(label, got, want):
    return (
        f"{label}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g}) "
        f"want=0x{want:08x} ({fp32_bits_to_float(want):.6g})"
    )


# -- tests ---------------------------------------------------------------


@cocotb.test()
async def test_single_rowblock(dut):
    await clk_setup(dut)
    await reset(dut)
    q1_blocks = 1
    weight_rows, acts, act_scales = random_rowblock(0xA11, q1_blocks)
    acts_beats = pack_acts_stream(acts, act_scales, q1_blocks)
    weights_beats = pack_weights_stream(weight_rows, q1_blocks)

    out_beats, tlast = await run_matmul(
        dut, acts_beats, weights_beats, q1_blocks, 1
    )
    assert tlast == [0, 0, 0, 1], f"TLAST pattern wrong: {tlast}"
    got = unpack_axis_results(out_beats)
    want = golden_rowblock(weight_rows, acts, act_scales)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)


@cocotb.test()
async def test_partial_lanes_zero_padded(dut):
    """Inactive lanes (M % ROWS != 0) get zero weight bits and emit 0.0."""
    await clk_setup(dut)
    await reset(dut)
    active_rows = 5
    q1_blocks = 16
    weight_rows, acts, act_scales = random_rowblock(
        0xB22, q1_blocks, active_rows=active_rows
    )
    acts_beats = pack_acts_stream(acts, act_scales, q1_blocks)
    weights_beats = pack_weights_stream(weight_rows, q1_blocks)

    out_beats, tlast = await run_matmul(
        dut, acts_beats, weights_beats, q1_blocks, 1
    )
    assert tlast[-1] == 1 and sum(tlast) == 1
    got = unpack_axis_results(out_beats)
    want = golden_rowblock(weight_rows, acts, act_scales)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)
        if lane >= active_rows:
            assert g == 0, f"inactive lane {lane} should be 0, got 0x{g:08x}"


@cocotb.test()
async def test_long_k_with_stream_stalls(dut):
    await clk_setup(dut)
    await reset(dut)
    q1_blocks = 16
    weight_rows, acts, act_scales = random_rowblock(0xC33, q1_blocks)
    acts_beats = pack_acts_stream(acts, act_scales, q1_blocks)
    weights_beats = pack_weights_stream(weight_rows, q1_blocks)

    out_beats, _ = await run_matmul(
        dut, acts_beats, weights_beats, q1_blocks, 1,
        acts_stall_rng=random.Random(0x5150),
        weights_stall_rng=random.Random(0x5151),
        output_ready_rng=random.Random(0x5152),
    )
    got = unpack_axis_results(out_beats)
    want = golden_rowblock(weight_rows, acts, act_scales)
    for lane, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == w, diff_msg(f"lane {lane}", g, w)


@cocotb.test()
async def test_multi_rowblock(dut):
    """Multi-rowblock matmul: acts sent once, weights for each rowblock.

    Critical v4 behaviour: every rowblock shares the same acts (loaded once
    into BRAM at the start), so this proves the BRAM broadcast works."""
    await clk_setup(dut)
    await reset(dut)
    q1_blocks = 4
    num_rowblocks = 3

    # One acts column, shared across all rowblocks.
    rng = random.Random(0xD00D)
    acts = [rng.randint(-128, 127) for _ in range(q1_blocks * Q1_BLOCK)]
    act_scales = [
        fp16_float_to_bits(0.001 + 0.0007 * ((i * 5 + 3) % 19))
        for i in range(q1_blocks * Q8_SUBBLOCKS)
    ]
    acts_beats = pack_acts_stream(acts, act_scales, q1_blocks)

    # Per-rowblock weights, golden uses the SAME acts/scales for every rb.
    weights_beats = []
    want_per_rowblock = []
    for rb in range(num_rowblocks):
        # New weights per rowblock (distinct seed); ignore the random acts
        # from random_rowblock since we use our shared one.
        weight_rows, _, _ = random_rowblock(0xE00 + rb, q1_blocks)
        weights_beats.extend(pack_weights_stream(weight_rows, q1_blocks))
        want_per_rowblock.append(
            golden_rowblock(weight_rows, acts, act_scales)
        )

    out_beats, tlast = await run_matmul(
        dut, acts_beats, weights_beats, q1_blocks, num_rowblocks,
        output_ready_rng=random.Random(0x6160),
    )

    expected_tlast = [0] * (num_rowblocks * 4)
    expected_tlast[-1] = 1
    assert tlast == expected_tlast, f"TLAST pattern wrong: {tlast}"

    got = unpack_axis_results(out_beats)
    for rb in range(num_rowblocks):
        for lane in range(ROWS):
            g = got[rb * ROWS + lane]
            w = want_per_rowblock[rb][lane]
            assert g == w, diff_msg(f"rb={rb} lane={lane}", g, w)
