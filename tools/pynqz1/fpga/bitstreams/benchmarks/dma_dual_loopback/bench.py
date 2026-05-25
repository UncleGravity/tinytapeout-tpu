#!/usr/bin/env python3
"""Dual-HP DMA loopback. Measures whether two PL streams scale.

Three measurement modes per transfer size:
    hp0    one DMA on HP0 alone (baseline, matches dma_loopback)
    hp1    one DMA on HP1 alone (proves HP1 plumbing)
    both   both DMAs concurrent (the question we actually care about)

If ``both`` ≈ ``hp0 + hp1``, the W1A8 plan can safely assume two independent
streams (weights on HP0, acts on HP1). If ``both`` saturates near a single
port's bandwidth, the DDR controller is the wall — adding HP ports won't help.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
from pynq import Overlay, allocate

DEFAULT_BITFILE = "/home/xilinx/overlays/dma_dual_loopback.bit"
DEFAULT_SIZES_MIB = (1, 4, 16, 32)
MIB = 1024 * 1024


def _alloc_pair(nbytes: int, seed: int):
    src = allocate(shape=(nbytes,), dtype=np.uint8)
    dst = allocate(shape=(nbytes,), dtype=np.uint8)
    # Distinct fill per pair so a HP0/HP1 wire crossing would be caught
    # by the byte-equality check, not silently passed by identical buffers.
    src[:] = (np.arange(nbytes, dtype=np.uint32) + seed).astype(np.uint8)
    return src, dst


def _kick(dma, src, dst) -> None:
    src.flush()
    dma.sendchannel.transfer(src)
    dma.recvchannel.transfer(dst)


def _wait(dma) -> None:
    dma.sendchannel.wait()
    dma.recvchannel.wait()


def _round_trip_single(dma, src, dst) -> float:
    src.flush()
    t0 = time.perf_counter()
    dma.sendchannel.transfer(src)
    dma.recvchannel.transfer(dst)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dt = time.perf_counter() - t0
    dst.invalidate()
    return dt


def _round_trip_both(dma0, src0, dst0, dma1, src1, dst1) -> float:
    # Pre-flush both sources so the timed region is just transfer + wait.
    src0.flush()
    src1.flush()
    t0 = time.perf_counter()
    # Issue all four channels before any wait — PYNQ DMA transfer() is
    # non-blocking, so this is what produces actual HP0/HP1 concurrency.
    dma0.sendchannel.transfer(src0)
    dma0.recvchannel.transfer(dst0)
    dma1.sendchannel.transfer(src1)
    dma1.recvchannel.transfer(dst1)
    dma0.sendchannel.wait()
    dma0.recvchannel.wait()
    dma1.sendchannel.wait()
    dma1.recvchannel.wait()
    dt = time.perf_counter() - t0
    dst0.invalidate()
    dst1.invalidate()
    return dt


def _verify_buffers(name: str, src, dst, nbytes: int) -> int:
    if np.array_equal(src, dst):
        return 0
    mismatches = np.flatnonzero(np.asarray(src) != np.asarray(dst))
    print(f"FAIL: {name} round-trip mismatch at {nbytes} bytes "
          f"(first mismatch offset {int(mismatches[0])})", file=sys.stderr)
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    overlay = Overlay(args.bitfile)
    dma0 = overlay.axi_dma_0
    dma1 = overlay.axi_dma_1

    src0, dst0 = _alloc_pair(args.nbytes, seed=0x1000)
    src1, dst1 = _alloc_pair(args.nbytes, seed=0x2000)
    try:
        dt0 = _round_trip_single(dma0, src0, dst0)
        rc = _verify_buffers("hp0", src0, dst0, args.nbytes)
        if rc:
            return rc

        dt1 = _round_trip_single(dma1, src1, dst1)
        rc = _verify_buffers("hp1", src1, dst1, args.nbytes)
        if rc:
            return rc

        # Re-fill so the concurrent pass is a real test, not a no-op.
        src0[:] = (np.arange(args.nbytes, dtype=np.uint32) + 0x3000).astype(np.uint8)
        src1[:] = (np.arange(args.nbytes, dtype=np.uint32) + 0x4000).astype(np.uint8)
        dst0[:] = 0
        dst1[:] = 0
        dt_both = _round_trip_both(dma0, src0, dst0, dma1, src1, dst1)
        rc = _verify_buffers("both/hp0", src0, dst0, args.nbytes) or \
             _verify_buffers("both/hp1", src1, dst1, args.nbytes)
        if rc:
            return rc

        mib = args.nbytes / MIB
        print(f"ok  hp0:   {mib:.1f} MiB  {dt0 * 1000:.2f} ms  "
              f"{2 * args.nbytes / dt0 / MIB:.0f} MiB/s round-trip")
        print(f"ok  hp1:   {mib:.1f} MiB  {dt1 * 1000:.2f} ms  "
              f"{2 * args.nbytes / dt1 / MIB:.0f} MiB/s round-trip")
        print(f"ok  both:  {mib:.1f} MiB  {dt_both * 1000:.2f} ms  "
              f"{4 * args.nbytes / dt_both / MIB:.0f} MiB/s aggregate")
        return 0
    finally:
        for buf in (src0, dst0, src1, dst1):
            buf.freebuffer()
        gc.collect()


def cmd_bench(args: argparse.Namespace) -> int:
    overlay = Overlay(args.bitfile)
    dma0 = overlay.axi_dma_0
    dma1 = overlay.axi_dma_1

    print(f"{'size':>10} {'hp0 MiB/s':>11} {'hp1 MiB/s':>11} "
          f"{'both MiB/s':>12} {'scale':>7}")
    for size_mib in args.sizes:
        nbytes = size_mib * MIB
        # CMA fragments fast on this board; force a sweep before each size
        # so freed buffers from the previous iteration actually return to the
        # pool, then skip cleanly if the contiguous region isn't available.
        gc.collect()
        try:
            src0, dst0 = _alloc_pair(nbytes, seed=0x1000)
            src1, dst1 = _alloc_pair(nbytes, seed=0x2000)
        except (MemoryError, RuntimeError) as exc:
            msg = str(exc).splitlines()[0][:60]
            print(f"{size_mib:>7d} MiB  skip (CMA exhausted: {msg})")
            continue
        try:
            # Warmup each path once.
            _round_trip_single(dma0, src0, dst0)
            _round_trip_single(dma1, src1, dst1)
            _round_trip_both(dma0, src0, dst0, dma1, src1, dst1)

            reps = 5
            t0 = t1 = tb = 0.0
            for _ in range(reps):
                t0 += _round_trip_single(dma0, src0, dst0)
                t1 += _round_trip_single(dma1, src1, dst1)
                tb += _round_trip_both(dma0, src0, dst0, dma1, src1, dst1)
            t0 /= reps
            t1 /= reps
            tb /= reps

            # Round-trip MiB/s (read + write counted) per port.
            bw0 = 2 * nbytes / t0 / MIB
            bw1 = 2 * nbytes / t1 / MIB
            # Concurrent: 4 transfers happened (2 per DMA), so 4x bytes.
            bw_both = 4 * nbytes / tb / MIB
            scale = bw_both / (bw0 + bw1) if (bw0 + bw1) > 0 else 0.0

            print(f"{size_mib:>7d} MiB {bw0:>11.1f} {bw1:>11.1f} "
                  f"{bw_both:>12.1f} {scale:>7.2f}")
        finally:
            for buf in (src0, dst0, src1, dst1):
                buf.freebuffer()
            gc.collect()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-HP DMA loopback bench")
    parser.add_argument("--bitfile", default=DEFAULT_BITFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="round-trip byte-equality on each path")
    v.add_argument("--bytes", dest="nbytes", type=int, default=4 * MIB)
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("bench", help="bandwidth sweep across sizes (MiB)")
    b.add_argument(
        "--sizes",
        type=lambda s: [int(x) for x in s.split(",") if x],
        default=list(DEFAULT_SIZES_MIB),
    )
    b.set_defaults(func=cmd_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
