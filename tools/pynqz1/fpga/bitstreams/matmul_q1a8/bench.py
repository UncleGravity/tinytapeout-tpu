#!/usr/bin/env python3
"""matmul_q1a8 bitstream sanity + perf test.

Loads the overlay, configures the AXI DMA and the q1a8_kernel_top register
file, streams a packed sub-block buffer through the DMA, polls done, reads
the fp32 result, and compares against the truncating-fp32 Python golden
(bit-exact mirror of the synthesizable arithmetic).

Validates the bitstream WITHOUT the daemon stack. Once `verify` passes,
the daemon-side PL kernel driver in board/kernels/pl/matmul_q1a8.py can
be written against known-good hardware.
"""

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

REG_ID            = 0x00
REG_VERSION       = 0x04
REG_CTRL          = 0x08
REG_STATUS        = 0x0C
REG_NUM_SUBBLOCKS = 0x10
REG_RESULT        = 0x14
REG_CYCLES        = 0x18

CTRL_START  = 1 << 0
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1

EXPECTED_ID      = 0xB05A_1000
EXPECTED_VERSION = 1


@dataclass
class KernelRun:
    result: int
    cycles: int
    timings_us: dict[str, int]


# -- Python golden (mirror of fp32_mul.v / fp32_add.v truncating arithmetic) --

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
    sign_small, _, mant_small   = (sb, eb, (1 << 23) | mb) if a_ge_b else (sa, ea, (1 << 23) | ma)
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


def golden_full_kernel(sub_blocks):
    acc = 0
    for wb, acts, ws_h, as_h in sub_blocks:
        sub_sum = sum(acts[i] if (wb >> i) & 1 else -acts[i] for i in range(32))
        ws_f32 = fp16_bits_to_fp32_bits(ws_h)
        as_f32 = fp16_bits_to_fp32_bits(as_h)
        ss_f32 = int_to_fp32_bits(sub_sum, width=14)
        combined = fp32_mul_trunc(ws_f32, as_f32)
        contrib = fp32_mul_trunc(combined, ss_f32)
        acc = fp32_add_trunc(acc, contrib)
    return acc


# -- Pack sub-block to bytes (matches axis_to_subblock.v) --------------------

def fp16_float_to_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def fp32_bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def random_sub_block(rng):
    return (
        rng.randint(0, 2**32 - 1),
        [rng.randint(-128, 127) for _ in range(32)],
        fp16_float_to_bits(rng.uniform(1e-3, 1.0)),
        fp16_float_to_bits(rng.uniform(1e-3, 1.0)),
    )


def pack_subblock_bytes(sb):
    """One sub-block -> 48 bytes (6 little-endian 64-bit beats)."""
    wb, acts, ws_h, as_h = sb
    parts = []
    parts.append(struct.pack("<II", wb & 0xFFFFFFFF, 0))                 # beat 0
    for i in range(0, 32, 8):                                            # beats 1-4
        parts.append(bytes(b & 0xFF for b in acts[i:i + 8]))
    parts.append(struct.pack("<HHI", ws_h, as_h, 0))                     # beat 5
    return b"".join(parts)


# -- Bench --------------------------------------------------------------------

def elapsed_us(start_ns):
    return max(0, (time.perf_counter_ns() - start_ns) // 1000)


def run_kernel(overlay, sub_blocks, buf):
    """Configure + DMA + start + poll; return result, cycles, and phase timing."""
    kernel = overlay.q1a8_kernel_top_0
    dma = overlay.axi_dma_0

    timings = {}
    total_start = time.perf_counter_ns()

    start = time.perf_counter_ns()
    packed = b"".join(pack_subblock_bytes(sb) for sb in sub_blocks)
    nbytes = len(packed)
    assert nbytes == 48 * len(sub_blocks)
    assert nbytes <= len(buf), f"buffer too small: {len(buf)} < {nbytes}"
    timings["pack_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    buf[:nbytes] = np.frombuffer(packed, dtype=np.uint8)
    buf.flush()
    timings["buffer_load_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    kernel.write(REG_NUM_SUBBLOCKS, len(sub_blocks))
    timings["kernel_setup_us"] = elapsed_us(start)

    # Arm DMA first, then start the kernel. The DMA may present the first
    # assembled sub-block and stall until the streamer becomes ready, but
    # PYNQ's transfer() call itself is non-blocking. This keeps most PS setup
    # overhead out of the kernel CYCLES register while avoiding a pre-start
    # dma.wait() deadlock.
    start = time.perf_counter_ns()
    dma.sendchannel.transfer(buf[:nbytes])
    timings["dma_start_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    kernel.write(REG_CTRL, CTRL_START)
    timings["kernel_start_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    dma.sendchannel.wait()
    timings["dma_wait_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    for _ in range(10000):
        status = kernel.read(REG_STATUS)
        if status & STATUS_DONE:
            break
    else:
        raise RuntimeError(f"kernel never reported done (status=0x{status:08x})")
    timings["poll_us"] = elapsed_us(start)

    start = time.perf_counter_ns()
    result = kernel.read(REG_RESULT)
    cycles = kernel.read(REG_CYCLES)
    timings["result_read_us"] = elapsed_us(start)
    timings["total_us"] = elapsed_us(total_start)

    return KernelRun(result=result, cycles=cycles, timings_us=timings)


def cmd_verify(args):
    overlay = Overlay(args.bitfile)
    kernel = overlay.q1a8_kernel_top_0

    got_id = kernel.read(REG_ID)
    if got_id != EXPECTED_ID:
        print(f"FAIL: ID got 0x{got_id:08x}, want 0x{EXPECTED_ID:08x}", file=sys.stderr)
        return 1
    print(f"ok   ID:      0x{got_id:08x}")

    got_ver = kernel.read(REG_VERSION)
    if got_ver != EXPECTED_VERSION:
        print(f"FAIL: VERSION got 0x{got_ver:08x}, want 0x{EXPECTED_VERSION:08x}", file=sys.stderr)
        return 1
    print(f"ok   VERSION: 0x{got_ver:08x}")

    idle_status = kernel.read(REG_STATUS)
    if idle_status != 0:
        print(f"warn idle STATUS non-zero: 0x{idle_status:08x}", file=sys.stderr)

    rng = random.Random(args.seed)
    sub_blocks = [random_sub_block(rng) for _ in range(args.num_subblocks)]

    buf = allocate(shape=(48 * args.num_subblocks,), dtype=np.uint8)
    try:
        run = run_kernel(overlay, sub_blocks, buf)
        result, cycles = run.result, run.cycles
        want = golden_full_kernel(sub_blocks)
        if result == want:
            print(f"ok   kernel: result=0x{result:08x} ({fp32_bits_to_float(result):.6g})"
                  f"  cycles={cycles}")
            return 0
        print(f"FAIL: result=0x{result:08x} ({fp32_bits_to_float(result):.6g})"
              f"  want=0x{want:08x} ({fp32_bits_to_float(want):.6g})", file=sys.stderr)
        return 1
    finally:
        buf.freebuffer()
        gc.collect()


def cmd_bench(args):
    overlay = Overlay(args.bitfile)
    rng = random.Random(0x42)
    sub_blocks = [random_sub_block(rng) for _ in range(args.num_subblocks)]
    want = golden_full_kernel(sub_blocks)

    buf = allocate(shape=(48 * args.num_subblocks,), dtype=np.uint8)
    try:
        # Warmup + sanity
        run = run_kernel(overlay, sub_blocks, buf)
        if run.result != want:
            print("FAIL: warmup mismatch", file=sys.stderr)
            return 1

        timing_totals = {}
        cycle_total = 0
        t0 = time.perf_counter()
        for _ in range(args.iters):
            run = run_kernel(overlay, sub_blocks, buf)
            if run.result != want:
                print("FAIL: mid-loop mismatch", file=sys.stderr)
                return 1
            cycle_total += run.cycles
            for key, value in run.timings_us.items():
                timing_totals[key] = timing_totals.get(key, 0) + value
        dt = time.perf_counter() - t0

        us_per_kernel = dt / args.iters * 1e6
        kernels_per_sec = args.iters / dt
        avg_cycles = cycle_total / args.iters
        pl_us = avg_cycles / 100.0  # 100 MHz fabric clock

        print(f"{args.iters} kernels, {args.num_subblocks} sub-blocks each (K={args.num_subblocks*32})")
        print(f"  wall:       {dt*1000:.1f} ms  ({us_per_kernel:.1f} us/kernel,"
              f" {kernels_per_sec:.0f} kernels/s)")
        print(f"  pl compute: {avg_cycles:.1f} cycles ({pl_us:.2f} us @ 100 MHz)")
        print(f"  ps overhead per kernel: {us_per_kernel - pl_us:.1f} us")
        print("  phase avg:")
        for key in (
            "pack_us",
            "buffer_load_us",
            "kernel_setup_us",
            "dma_start_us",
            "kernel_start_us",
            "dma_wait_us",
            "poll_us",
            "result_read_us",
            "total_us",
        ):
            print(f"    {key:16s} {timing_totals.get(key, 0) / args.iters:.1f} us")
        return 0
    finally:
        buf.freebuffer()
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description="matmul_q1a8 bitstream tests")
    parser.add_argument("--bitfile", default=DEFAULT_BITFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="single kernel correctness check")
    v.add_argument("--num-subblocks", type=int, default=64)
    v.add_argument("--seed", type=int, default=0xCAFE)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="back-to-back kernel throughput")
    b.add_argument("--num-subblocks", type=int, default=64)
    b.add_argument("--iters", type=int, default=1000)
    b.set_defaults(func=cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
