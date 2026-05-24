"""``python -m board.daemon`` entrypoint.

Constructs the kernel registry, allocator, and overlay, then serves
RPC frames until interrupted.
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
    parser = argparse.ArgumentParser(description="PYNQ-Z1 Bonsai board daemon")
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
    return parser.parse_args(argv)


def make_runtime(args: argparse.Namespace) -> Runtime:
    total = args.heap_mib * MIB
    slab = args.slab_mib * MIB
    if args.allocator == "fake":
        allocator = TensorAllocator(fake_slabs(total, slab))
        overlay = None
    else:
        overlay = load_overlay(args.overlay)
        allocator = TensorAllocator(pynq_slabs(total, slab))

    registry = KernelRegistry()
    ps_native.register_all(registry)
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
        print(
            f"bonsaid listening on {host}:{port} "
            f"allocator={args.allocator} heap={args.heap_mib}MiB slab={args.slab_mib}MiB",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("bonsaid stopping", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
