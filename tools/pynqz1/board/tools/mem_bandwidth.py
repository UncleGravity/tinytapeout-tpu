#!/usr/bin/env python3
"""Small PYNQ-Z1 DDR bandwidth benchmark.

This measures PS-side access to DDR-backed PYNQ buffers. It does not measure
PL AXI HP/DMA bandwidth; that needs a loopback or accelerator overlay.
"""

import argparse
import gc
import time

import numpy as np

MIB = 1024 * 1024


def parse_sizes(text):
    sizes = []
    for item in text.split(","):
        item = item.strip()
        if item:
            sizes.append(int(item))
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    return sizes


def load_overlay(name):
    if name == "none":
        return None
    if name == "base":
        from pynq.overlays.base import BaseOverlay

        return BaseOverlay("base.bit")

    from pynq import Overlay

    return Overlay(name)


def run_case(name, traffic_bytes, fn, min_seconds):
    for _ in range(3):
        fn()

    reps = 0
    start = time.perf_counter()
    while True:
        fn()
        reps += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds and reps >= 3:
            break

    mib_s = (traffic_bytes * reps) / elapsed / MIB
    return reps, elapsed, mib_s


def print_result(size_mib, name, traffic_mib, reps, seconds, mib_s):
    print(
        f"{size_mib:>8}  {name:<14}  "
        f"{traffic_mib:>11.1f}  {reps:>6}  {seconds:>8.3f}  {mib_s:>10.1f}"
    )


def bench_size(size_mib, min_seconds):
    from pynq import allocate

    nbytes = size_mib * MIB
    src = allocate(shape=(nbytes,), dtype=np.uint8)
    dst = allocate(shape=(nbytes,), dtype=np.uint8)

    try:
        print(
            f"\nsize={size_mib} MiB "
            f"src_pa=0x{src.physical_address:x} dst_pa=0x{dst.physical_address:x} "
            f"coherent={getattr(src, 'coherent', 'unknown')}"
        )

        src.fill(0x5A)
        dst.fill(0)
        src.flush()
        dst.flush()

        cases = [
            ("fill", nbytes, lambda: src.fill(0xA5)),
            ("sum", nbytes, lambda: int(src.sum(dtype=np.uint64))),
            ("copyto", 2 * nbytes, lambda: np.copyto(dst, src)),
            ("xor_in_place", 2 * nbytes, lambda: np.bitwise_xor(src, 0xFF, out=src)),
            ("flush", nbytes, lambda: src.flush()),
            ("invalidate", nbytes, lambda: dst.invalidate()),
        ]

        for name, traffic, fn in cases:
            reps, seconds, mib_s = run_case(name, traffic, fn, min_seconds)
            print_result(size_mib, name, traffic / MIB, reps, seconds, mib_s)
    finally:
        src.freebuffer()
        dst.freebuffer()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overlay",
        default="base",
        help="'base' for packaged base.bit, 'none' for already-loaded overlay, or a bitfile path",
    )
    parser.add_argument(
        "--sizes-mib",
        type=parse_sizes,
        default=parse_sizes("1,4,16,32,64"),
        help="comma-separated buffer sizes in MiB",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.5,
        help="minimum time per benchmark case",
    )
    args = parser.parse_args()

    print("PYNQ-Z1 DDR bandwidth benchmark")
    print("NOTE: this is PS-side DDR access, not PL AXI DMA bandwidth.")

    overlay = load_overlay(args.overlay)
    if overlay is not None:
        print(f"overlay={overlay.bitfile_name}")

    print("\nsize_mib  case            traffic_mib    reps   seconds     MiB/s")
    print("--------  --------------  -----------  ------  --------  ---------")

    for size_mib in args.sizes_mib:
        bench_size(size_mib, args.min_seconds)


if __name__ == "__main__":
    main()
