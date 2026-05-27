"""``python -m board.daemon`` entrypoint.

Constructs the kernel registry, allocator, and (optionally) the PL
overlay, then serves RPC frames until interrupted.

PL kernels register after PS kernels — name collisions like ``GOP_COPY``
override the PS implementation. This is the swap-point that lets a
bitstream graduate from "loopback for testing" to "real W1A8 matmul"
without daemon code changes.
"""

from __future__ import annotations

import argparse
import sys

from board.daemon.runtime import Runtime
from board.daemon.server import BonsaiRpcServer
from board.kernels.ps import native as ps_native
from board.kernels.registry import KernelRegistry
from board.memory.allocator import TensorAllocator
from board.memory.slabs import MIB, fake_slabs, pynq_slabs
from proto.ops import DEFAULT_PORT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PYNQ-Z1 board daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allocator", choices=["fake", "pynq"], default="fake")
    parser.add_argument("--heap-mib", type=int, default=64)
    parser.add_argument("--slab-mib", type=int, default=32)
    parser.add_argument(
        "--overlay",
        default="base",
        help="'base', 'none', or a bitfile path (pynq allocator only)",
    )
    parser.add_argument("--overlay-id", default="fake-local")
    parser.add_argument(
        "--bitfile",
        default=None,
        help="path to a PL bitstream; if set, board.kernels.pl.register_all "
             "is called against the loaded overlay (overrides matching PS kernels)",
    )
    return parser.parse_args(argv)


def make_runtime(args: argparse.Namespace) -> Runtime:
    total = args.heap_mib * MIB
    slab = args.slab_mib * MIB
    if args.allocator == "fake":
        allocator = TensorAllocator(fake_slabs(total, slab))
        overlay = None
    else:
        overlay = load_overlay(args.bitfile or args.overlay)
        allocator = TensorAllocator(pynq_slabs(total, slab))

    registry = KernelRegistry()
    ps_native.register_all(registry)

    if args.bitfile is not None and overlay is not None:
        from board.kernels import pl
        pl.register_all(registry, overlay)

    return Runtime(allocator, registry, args.overlay_id, overlay)


def load_overlay(path: str) -> object | None:
    if path == "none":
        return None
    if path == "base":
        from pynq.overlays.base import BaseOverlay

        return BaseOverlay("base.bit")
    from pynq import Overlay

    return Overlay(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime = make_runtime(args)
    with BonsaiRpcServer((args.host, args.port), runtime) as server:
        host, port = server.server_address
        ops = ", ".join(runtime.registry.names())
        print(
            f"pynqd listening on {host}:{port} "
            f"allocator={args.allocator} heap={args.heap_mib}MiB slab={args.slab_mib}MiB "
            f"ops=[{ops}]",
            flush=True,
        )
        # debug: probe q1a8 kernel ID at startup; helps diagnose MMIO corruption.
        if runtime.overlay is not None and hasattr(runtime.overlay, "q1a8_kernel_top_0"):
            kid = runtime.overlay.q1a8_kernel_top_0.read(0x00)
            kver = runtime.overlay.q1a8_kernel_top_0.read(0x04)
            print(f"q1a8 startup MMIO: ID={kid:#010x} VER={kver:#010x}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("pynqd stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
