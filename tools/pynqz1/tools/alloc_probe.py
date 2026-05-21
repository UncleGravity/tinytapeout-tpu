#!/usr/bin/env python3
"""Probe PYNQ-Z1 DDR allocation limits for accelerator buffers.

This measures what PYNQ's `allocate()` can actually reserve, not just the
board's nominal DDR size. On Zynq/PYNQ, large physically-addressable buffers
usually come from CMA, so the useful limit can be much smaller than 512 MiB.
"""

import argparse
import gc
import os
import time

import numpy as np
from pynq import Overlay, allocate


MIB = 1024 * 1024


def read_meminfo():
    wanted = {
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "CmaTotal",
        "CmaFree",
    }
    out = {}
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                key, rest = line.split(":", 1)
                if key in wanted:
                    parts = rest.strip().split()
                    out[key] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def print_meminfo(label):
    info = read_meminfo()
    print(f"\n{label}")
    for key in [
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "CmaTotal",
        "CmaFree",
    ]:
        if key in info:
            print(f"  {key:<13} {info[key] / MIB:8.1f} MiB")


def load_overlay(path):
    if path == "none":
        return None
    if path == "base":
        from pynq.overlays.base import BaseOverlay

        return BaseOverlay("base.bit")
    return Overlay(path)


def try_alloc(size_mib, touch):
    nbytes = size_mib * MIB
    buf = allocate(shape=(nbytes,), dtype=np.uint8)
    try:
        if touch:
            # Touch one byte per page and the final byte to force backing pages
            # and catch late failures before reporting success.
            page = os.sysconf("SC_PAGE_SIZE")
            buf[0:nbytes:page] = 0xA5
            buf[nbytes - 1] = 0x5A
            buf.flush()
        return buf
    except Exception:
        buf.freebuffer()
        raise


def find_max_single(max_mib, step_mib, touch):
    print("\nSingle-buffer probe")
    last_ok = 0
    last_err = None

    size = step_mib
    while size <= max_mib:
        t0 = time.perf_counter()
        try:
            buf = try_alloc(size, touch)
            dt = time.perf_counter() - t0
            print(
                f"  ok   {size:5d} MiB  "
                f"pa=0x{buf.physical_address:08x}  {dt:6.3f}s"
            )
            last_ok = size
            buf.freebuffer()
            gc.collect()
        except Exception as e:
            last_err = e
            print(f"  fail {size:5d} MiB  {type(e).__name__}: {e}")
            break
        size += step_mib

    lo = last_ok
    hi = min(size, max_mib)
    if lo < hi:
        print(f"\nBinary-searching max single allocation between {lo} and {hi} MiB")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        t0 = time.perf_counter()
        try:
            buf = try_alloc(mid, touch)
            dt = time.perf_counter() - t0
            print(
                f"  ok   {mid:5d} MiB  "
                f"pa=0x{buf.physical_address:08x}  {dt:6.3f}s"
            )
            lo = mid
            buf.freebuffer()
            gc.collect()
        except Exception as e:
            last_err = e
            print(f"  fail {mid:5d} MiB  {type(e).__name__}: {e}")
            hi = mid

    print(f"\nmax_single_mib={lo}")
    if last_err is not None:
        print(f"first_error={type(last_err).__name__}: {last_err}")
    return lo


def find_chunked_total(chunk_mib, max_chunks, touch):
    print("\nChunked allocation probe")
    bufs = []
    try:
        for i in range(max_chunks):
            t0 = time.perf_counter()
            try:
                buf = try_alloc(chunk_mib, touch)
            except Exception as e:
                print(
                    f"  fail chunk={i:3d} total={len(bufs) * chunk_mib:5d} MiB  "
                    f"{type(e).__name__}: {e}"
                )
                break
            dt = time.perf_counter() - t0
            bufs.append(buf)
            print(
                f"  ok   chunk={i:3d} total={len(bufs) * chunk_mib:5d} MiB  "
                f"pa=0x{buf.physical_address:08x}  {dt:6.3f}s"
            )
        print(f"\nmax_chunked_total_mib={len(bufs) * chunk_mib}")
    finally:
        for buf in bufs:
            buf.freebuffer()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overlay",
        default="base",
        help="'base', 'none', or a bitfile path. Use 'none' if overlay is already loaded.",
    )
    parser.add_argument("--max-mib", type=int, default=448)
    parser.add_argument("--step-mib", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=32)
    parser.add_argument(
        "--no-touch",
        action="store_true",
        help="Allocate without writing/flushing pages. Faster but less realistic.",
    )
    args = parser.parse_args()

    print("PYNQ allocation probe")
    print_meminfo("Before overlay")

    overlay = load_overlay(args.overlay)
    if overlay is not None:
        print(f"\noverlay={overlay.bitfile_name}")

    print_meminfo("Before allocation")
    touch = not args.no_touch
    find_max_single(args.max_mib, args.step_mib, touch)
    print_meminfo("After single-buffer probe")
    find_chunked_total(args.chunk_mib, args.max_chunks, touch)
    print_meminfo("After chunked probe")


if __name__ == "__main__":
    main()
