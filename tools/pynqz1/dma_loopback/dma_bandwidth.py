#!/usr/bin/env python3
import argparse
import time

import numpy as np
from pynq import Overlay, allocate


MIB = 1024 * 1024


def parse_sizes(text):
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def time_transfer(dma, src, dst, min_seconds):
    src.flush()

    reps = 0
    start = time.perf_counter()
    while True:
        dma.recvchannel.transfer(dst)
        dma.sendchannel.transfer(src)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        reps += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds and reps >= 2:
            dst.invalidate()
            return reps, elapsed


def bench_size(dma, size_mib, min_seconds, verify):
    nbytes = size_mib * MIB
    src = allocate(shape=(nbytes,), dtype=np.uint8)
    dst = allocate(shape=(nbytes,), dtype=np.uint8)

    try:
        src[:] = np.arange(nbytes, dtype=np.uint8)
        dst[:] = 0
        src.flush()
        dst.flush()

        dma.recvchannel.transfer(dst)
        dma.sendchannel.transfer(src)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
        dst.invalidate()

        if verify and not np.array_equal(src, dst):
            raise RuntimeError(f"DMA loopback mismatch for {size_mib} MiB")

        reps, seconds = time_transfer(dma, src, dst, min_seconds)
        if verify and not np.array_equal(src, dst):
            raise RuntimeError(f"DMA loopback mismatch after timing for {size_mib} MiB")

        payload_mib_s = (nbytes * reps) / seconds / MIB
        ddr_mib_s = 2.0 * payload_mib_s
        print(
            f"{size_mib:>8}  {reps:>6}  {seconds:>8.3f}  "
            f"{payload_mib_s:>15.1f}  {ddr_mib_s:>17.1f}"
        )
    finally:
        src.freebuffer()
        dst.freebuffer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit", default="/home/xilinx/dma_loopback.bit")
    parser.add_argument("--sizes-mib", type=parse_sizes, default=parse_sizes("1,4,16,32"))
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    overlay = Overlay(args.bit)
    dma = overlay.axi_dma_0

    print(f"overlay={overlay.bitfile_name}")
    print("size_mib    reps   seconds  payload_MiB/s  ddr_traffic_MiB/s")
    print("--------  ------  --------  -------------  -----------------")
    for size_mib in args.sizes_mib:
        bench_size(dma, size_mib, args.min_seconds, not args.no_verify)


if __name__ == "__main__":
    main()
