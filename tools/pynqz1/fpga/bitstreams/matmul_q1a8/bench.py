#!/usr/bin/env python3
"""matmul_q1a8 rowblock bitstream sanity + perf test."""

from __future__ import annotations

import argparse
import gc
import random
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
from pynq import Overlay, allocate

DEFAULT_BITFILE = "/home/xilinx/overlays/matmul_q1a8.bit"
ROWS = 8
Q1_BLOCK = 128
Q8_BLOCK = 32

REG_ID = 0x00
REG_VERSION = 0x04
REG_CTRL = 0x08
REG_STATUS = 0x0C
REG_NUM_Q1_BLOCKS = 0x10
REG_NUM_ROWBLOCKS = 0x14
REG_CYCLES = 0x18
REG_ROWS = 0x1C

CTRL_START = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID = 0xB05A_2000
EXPECTED_VERSION = 3


@dataclass
class KernelRun:
    results: list[int]
    cycles: int
    timings_us: dict[str, int]


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
    """fp32 results for one rowblock (ROWS lanes)."""
    out = []
    for row in range(ROWS):
        acc = 0
        for q1_index, (ws_h, bits128) in enumerate(weight_rows[row]):
            for q8_local in range(0, Q1_BLOCK, Q8_BLOCK):
                q8_index = q1_index * 4 + q8_local // Q8_BLOCK
                act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
                wb = (bits128 >> q8_local) & 0xFFFF_FFFF
                acc = fp32_add_trunc(acc, contribution(wb, act_block, ws_h, act_scales[q8_index]))
        out.append(acc)
    return out


def random_rowblock(seed, q1_blocks, active_rows=ROWS):
    """Random weight rows + acts/scales for one rowblock. Lanes >= active_rows
    get zero-padded weight bits and scales (matching the host packer's
    behaviour for a partial last rowblock)."""
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
        for i in range(q1_blocks * 4)
    ]
    return weight_rows, acts, act_scales


def pack_rowblock(weight_rows, acts, act_scales, q1_blocks):
    """Pack one rowblock's beats into u64 words."""
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
            q8_index = q1_index * 4 + q8_local // Q8_BLOCK
            act_block = acts[q8_index * Q8_BLOCK : (q8_index + 1) * Q8_BLOCK]
            for beat in range(4):
                word = 0
                for byte in range(8):
                    word |= (act_block[beat * 8 + byte] & 0xFF) << (byte * 8)
                beats.append(word)
            beats.append(act_scales[q8_index])

            for beat in range(wbits_beats):
                word = 0
                for local in range(2):
                    lane = beat * 2 + local
                    bits = (weight_rows[lane][q1_index][1] >> q8_local) & 0xFFFF_FFFF if lane < ROWS else 0
                    word |= bits << (local * 32)
                beats.append(word)
    return beats


def build_matmul(seed, num_rowblocks, q1_blocks, last_active_rows=ROWS):
    """Build a multi-rowblock matmul. Returns (stream_bytes, expected_results).
    The final rowblock is zero-padded down to last_active_rows."""
    rng = random.Random(seed)
    all_beats = []
    expected = []
    for rb in range(num_rowblocks):
        active = last_active_rows if rb == num_rowblocks - 1 else ROWS
        wr, ac, sc = random_rowblock(rng.randint(0, 1 << 30), q1_blocks,
                                     active_rows=active)
        all_beats.extend(pack_rowblock(wr, ac, sc, q1_blocks))
        expected.extend(golden_rowblock(wr, ac, sc))
    return b"".join(struct.pack("<Q", w) for w in all_beats), expected


def elapsed_us(start_ns):
    return max(0, (time.perf_counter_ns() - start_ns) // 1000)


def run_kernel(overlay, packed, num_rowblocks, q1_blocks, send_buf, recv_buf):
    """Drive one full matmul: NUM_ROWBLOCKS rowblocks via S2MM result DMA."""
    kernel = overlay.q1a8_kernel_top_0
    dma = overlay.axi_dma_0
    timings = {}
    total_start = time.perf_counter_ns()

    stream_nbytes = len(packed)
    result_nbytes = num_rowblocks * ROWS * 4
    assert stream_nbytes <= len(send_buf), "send buf too small"
    assert result_nbytes <= len(recv_buf), "recv buf too small"

    start = time.perf_counter_ns()
    send_buf[:stream_nbytes] = np.frombuffer(packed, dtype=np.uint8)
    send_buf.flush()
    timings["buffer_load_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    kernel.write(REG_NUM_Q1_BLOCKS, q1_blocks)
    kernel.write(REG_NUM_ROWBLOCKS, num_rowblocks)
    timings["kernel_setup_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    dma.recvchannel.transfer(recv_buf[:result_nbytes])
    timings["recv_start_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    kernel.write(REG_CTRL, CTRL_START)
    timings["kernel_start_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    dma.sendchannel.transfer(send_buf[:stream_nbytes])
    timings["send_start_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    dma.sendchannel.wait()
    timings["send_wait_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    dma.recvchannel.wait()
    timings["recv_wait_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    for _ in range(10000):
        status = kernel.read(REG_STATUS)
        if status & STATUS_DONE:
            break
    else:
        raise RuntimeError(f"kernel never reported done (status=0x{status:08x})")
    timings["poll_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    recv_buf.invalidate()
    result_words = recv_buf[:result_nbytes].view(np.uint32)
    results = [int(w) & 0xFFFF_FFFF for w in result_words]
    cycles = int(kernel.read(REG_CYCLES))
    timings["result_read_us"] = elapsed_us(start)
    timings["total_us"] = elapsed_us(total_start)
    return KernelRun(results=results, cycles=cycles, timings_us=timings)


def check_id(kernel):
    got_id = kernel.read(REG_ID)
    if got_id != EXPECTED_ID:
        print(f"FAIL: ID got 0x{got_id:08x}, want 0x{EXPECTED_ID:08x}", file=sys.stderr)
        return False
    print(f"ok   ID:      0x{got_id:08x}")

    got_ver = kernel.read(REG_VERSION)
    if got_ver != EXPECTED_VERSION:
        print(f"FAIL: VERSION got 0x{got_ver:08x}, want 0x{EXPECTED_VERSION:08x}", file=sys.stderr)
        return False
    print(f"ok   VERSION: 0x{got_ver:08x}")

    got_rows = kernel.read(REG_ROWS)
    if got_rows != ROWS:
        print(f"FAIL: ROWS got {got_rows}, want {ROWS}", file=sys.stderr)
        return False
    print(f"ok   ROWS:    {got_rows}")
    return True


def _allocate_buffers(packed_nbytes, result_nbytes):
    send_buf = allocate(shape=(packed_nbytes,), dtype=np.uint8)
    recv_buf = allocate(shape=(result_nbytes,), dtype=np.uint8)
    return send_buf, recv_buf


def cmd_verify(args):
    overlay = Overlay(args.bitfile)
    if not check_id(overlay.q1a8_kernel_top_0):
        return 1

    packed, want = build_matmul(args.seed, args.rowblocks, args.q1_blocks,
                                last_active_rows=args.last_rows)
    result_nbytes = args.rowblocks * ROWS * 4
    send_buf, recv_buf = _allocate_buffers(len(packed), result_nbytes)
    try:
        run = run_kernel(overlay, packed, args.rowblocks, args.q1_blocks,
                         send_buf, recv_buf)
        for idx, (got, exp) in enumerate(zip(run.results, want, strict=True)):
            if got != exp:
                print(
                    f"FAIL idx {idx}: got=0x{got:08x} ({fp32_bits_to_float(got):.6g}) "
                    f"want=0x{exp:08x} ({fp32_bits_to_float(exp):.6g})",
                    file=sys.stderr,
                )
                return 1
        m_rows = args.rowblocks * ROWS - (ROWS - args.last_rows)
        print(f"ok   matmul: M={m_rows} K={args.q1_blocks * Q1_BLOCK} "
              f"rowblocks={args.rowblocks} cycles={run.cycles}")
        return 0
    finally:
        send_buf.freebuffer()
        recv_buf.freebuffer()
        gc.collect()


def cmd_bench(args):
    overlay = Overlay(args.bitfile)
    if not check_id(overlay.q1a8_kernel_top_0):
        return 1

    packed, want = build_matmul(0x42, args.rowblocks, args.q1_blocks)
    result_nbytes = args.rowblocks * ROWS * 4
    send_buf, recv_buf = _allocate_buffers(len(packed), result_nbytes)
    try:
        warmup = run_kernel(overlay, packed, args.rowblocks, args.q1_blocks,
                            send_buf, recv_buf)
        if warmup.results != want:
            print("FAIL: warmup mismatch", file=sys.stderr)
            return 1

        timing_totals = {}
        cycle_total = 0
        t0 = time.perf_counter()
        for _ in range(args.iters):
            run = run_kernel(overlay, packed, args.rowblocks, args.q1_blocks,
                             send_buf, recv_buf)
            if run.results != want:
                print("FAIL: mid-loop mismatch", file=sys.stderr)
                return 1
            cycle_total += run.cycles
            for key, value in run.timings_us.items():
                timing_totals[key] = timing_totals.get(key, 0) + value
        dt = time.perf_counter() - t0

        m_rows = args.rowblocks * ROWS
        k = args.q1_blocks * Q1_BLOCK
        us_per_matmul = dt / args.iters * 1e6
        us_per_rowblock = us_per_matmul / args.rowblocks
        avg_cycles = cycle_total / args.iters
        pl_us = avg_cycles / 100.0
        macs_per_matmul = m_rows * k
        bytes_in_per_matmul = len(packed)
        print(
            f"{args.iters} matmuls, M={m_rows}, K={k}, rowblocks={args.rowblocks}, "
            f"stream={len(packed)} B"
        )
        print(f"  wall:        {dt*1000:.1f} ms ({us_per_matmul:.1f} us/matmul, "
              f"{us_per_rowblock:.1f} us/rowblock)")
        print(f"  pl compute:  {avg_cycles:.0f} cycles ({pl_us:.1f} us @ 100 MHz)")
        print(f"  ps overhead: {us_per_matmul - pl_us:.1f} us/matmul")
        print(f"  throughput:  {macs_per_matmul * args.iters / dt / 1e6:.1f} MMAC/s "
              f"({bytes_in_per_matmul * args.iters / dt / 1e6:.1f} MB/s in)")
        print("  phase avg (us/matmul):")
        for key in (
            "buffer_load_us",
            "kernel_setup_us",
            "recv_start_us",
            "kernel_start_us",
            "send_start_us",
            "send_wait_us",
            "recv_wait_us",
            "poll_us",
            "result_read_us",
            "total_us",
        ):
            print(f"    {key:16s} {timing_totals.get(key, 0) / args.iters:.1f}")
        return 0
    finally:
        send_buf.freebuffer()
        recv_buf.freebuffer()
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description="matmul_q1a8 bitstream tests")
    parser.add_argument("--bitfile", default=DEFAULT_BITFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="end-to-end correctness check")
    v.add_argument("--rowblocks", type=int, default=1, help="number of rowblocks")
    v.add_argument("--last-rows", type=int, default=ROWS,
                   choices=range(1, ROWS + 1),
                   help="active lanes in the last rowblock (zero-pad remainder)")
    v.add_argument("--q1-blocks", type=int, default=16)
    v.add_argument("--seed", type=int, default=0xCAFE)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="full matmul throughput")
    b.add_argument("--rowblocks", type=int, default=32,
                   help="rowblocks per matmul (M = rowblocks * 8)")
    b.add_argument("--q1-blocks", type=int, default=16,
                   help="Q1 blocks per row (K = q1_blocks * 128)")
    b.add_argument("--iters", type=int, default=200)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
