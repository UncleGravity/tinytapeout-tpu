#!/usr/bin/env python3
"""DMA loopback sanity + bandwidth probe. Runs on the board.

Two subcommands:
    verify          One round-trip at a fixed size; asserts byte equality.
    bench           Sweep transfer sizes, report MB/s for each.

Uses the PYNQ Overlay API directly — no daemon, no kernel registry. The
single purpose is to validate the bitstream and the AXI HP DMA path
without anything else in the way.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
from pynq import Overlay, allocate

DEFAULT_BITFILE = "/home/xilinx/overlays/dma_loopback.bit"
DEFAULT_SIZES_MIB = (1, 4, 16, 64)
MIB = 1024 * 1024


def _round_trip(dma, src, dst) -> float:
    src.flush()
    t0 = time.perf_counter()
    dma.sendchannel.transfer(src)
    dma.recvchannel.transfer(dst)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    dt = time.perf_counter() - t0
    dst.invalidate()
    return dt


def _alloc_pair(nbytes: int):
    src = allocate(shape=(nbytes,), dtype=np.uint8)
    dst = allocate(shape=(nbytes,), dtype=np.uint8)
    # Deterministic pattern so we can spot misalignment in failure logs.
    src[:] = np.arange(nbytes, dtype=np.uint8)
    return src, dst


def cmd_verify(args: argparse.Namespace) -> int:
    overlay = Overlay(args.bitfile)
    src, dst = _alloc_pair(args.nbytes)
    try:
        dt = _round_trip(overlay.axi_dma_0, src, dst)
        if not np.array_equal(src, dst):
            print(f"FAIL: round-trip mismatch at {args.nbytes} bytes", file=sys.stderr)
            mismatches = np.flatnonzero(src != dst)
            print(f"  first mismatch at offset {int(mismatches[0])}", file=sys.stderr)
            return 1
        print(f"ok  {args.nbytes / MIB:.1f} MiB  {dt * 1000:.2f} ms  "
              f"{2 * args.nbytes / dt / MIB:.0f} MiB/s round-trip")
        return 0
    finally:
        src.freebuffer()
        dst.freebuffer()
        gc.collect()


def cmd_bench(args: argparse.Namespace) -> int:
    overlay = Overlay(args.bitfile)
    print(f"{'size':>10} {'rt_ms':>8} {'one-way MiB/s':>14} {'round-trip MiB/s':>17}")
    for size_mib in args.sizes:
        nbytes = size_mib * MIB
        src, dst = _alloc_pair(nbytes)
        try:
            # Warmup, then average a few reps.
            _round_trip(overlay.axi_dma_0, src, dst)
            reps = 5
            total = 0.0
            for _ in range(reps):
                total += _round_trip(overlay.axi_dma_0, src, dst)
            dt = total / reps
            if not np.array_equal(src, dst):
                print(f"FAIL: {size_mib} MiB mismatch", file=sys.stderr)
                return 1
            print(f"{size_mib:>7d} MiB {dt * 1000:>8.2f} "
                  f"{nbytes / dt / MIB:>14.1f} {2 * nbytes / dt / MIB:>17.1f}")
        finally:
            src.freebuffer()
            dst.freebuffer()
            gc.collect()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMA loopback sanity + benchmark")
    parser.add_argument("--bitfile", default=DEFAULT_BITFILE)
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="single round-trip byte-equality check")
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
