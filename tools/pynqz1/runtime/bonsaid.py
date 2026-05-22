from __future__ import annotations

import argparse
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.allocator import (  # noqa: E402
    MIB,
    AllocatorError,
    TensorAllocator,
    fake_allocator,
    pynq_allocator,
)
from runtime.bonsai_rpc import ProtocolError, recv_message, send_message  # noqa: E402
from runtime.graph import run_graph  # noqa: E402


ABI_VERSION = 1
DEFAULT_PORT = 50055


class BonsaiRuntime:
    def __init__(
        self,
        allocator: TensorAllocator,
        overlay_id: str,
        overlay: object | None = None,
    ):
        self.allocator = allocator
        self.overlay_id = overlay_id
        self.overlay = overlay
        self._lock = threading.Lock()

    def dispatch(
        self, metadata: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        op = str(metadata.get("op", ""))
        if not op:
            raise AllocatorError("invalid_request", "missing op")

        if op == "HELLO":
            return self._hello(), b""

        if op == "MEMORY":
            return {"memory": self.allocator.memory_info()}, b""

        if op == "ALLOC_TENSOR":
            with self._lock:
                record = self.allocator.allocate(
                    _required_int(metadata, "nbytes"),
                    shape=_optional_list(metadata, "shape"),
                    dtype=str(metadata.get("dtype", "u8")),
                    usage=str(metadata.get("usage", "tensor")),
                    layout=str(metadata.get("layout", "raw")),
                    alignment=int(metadata.get("alignment", 64)),
                )
                return {"tensor": self.allocator.describe(record.handle)}, b""

        if op == "UPLOAD_TENSOR":
            handle = _required_int(metadata, "handle")
            offset = int(metadata.get("offset", 0))
            with self._lock:
                self.allocator.write(handle, offset, payload)
                tensor = self.allocator.describe(handle)
            return {"tensor": tensor, "written": len(payload)}, b""

        if op == "DOWNLOAD_TENSOR":
            handle = _required_int(metadata, "handle")
            offset = int(metadata.get("offset", 0))
            size = _required_int(metadata, "size")
            with self._lock:
                data = self.allocator.read(handle, offset, size)
                tensor = self.allocator.describe(handle)
            return {"tensor": tensor, "read": len(data)}, data

        if op == "FREE_TENSOR":
            handle = _required_int(metadata, "handle")
            with self._lock:
                record = self.allocator.free(handle)
                memory = self.allocator.memory_info()
            return {
                "freed": {
                    "handle": record.handle,
                    "nbytes": record.nbytes,
                },
                "memory": memory,
            }, b""

        if op == "RUN_GRAPH":
            with self._lock:
                return run_graph(self.allocator, metadata), b""

        raise AllocatorError("unsupported_op", f"unsupported op {op}")

    def close(self) -> None:
        self.allocator.close()

    def _hello(self) -> dict[str, Any]:
        return {
            "abi_version": ABI_VERSION,
            "server": "bonsaid",
            "overlay_id": self.overlay_id,
            "memory": self.allocator.memory_info(),
            "capabilities": [
                "ALLOC_TENSOR",
                "UPLOAD_TENSOR",
                "DOWNLOAD_TENSOR",
                "FREE_TENSOR",
                "RUN_GRAPH",
            ],
            "graph_ops": ["COPY"],
        }


class BonsaiRpcServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, runtime: BonsaiRuntime):
        super().__init__(server_address, BonsaiRequestHandler)
        self.runtime = runtime

    def server_close(self) -> None:
        try:
            self.runtime.close()
        finally:
            super().server_close()


class BonsaiRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        runtime: BonsaiRuntime = self.server.runtime
        while True:
            try:
                metadata, payload = recv_message(self.request)
            except EOFError:
                return
            except ProtocolError as exc:
                _send_error(self.request, None, "protocol_error", str(exc))
                return

            request_id = metadata.get("id")
            try:
                result, response_payload = runtime.dispatch(metadata, payload)
            except AllocatorError as exc:
                _send_error(self.request, request_id, exc.code, str(exc))
                continue
            except (TypeError, ValueError) as exc:
                _send_error(self.request, request_id, "invalid_request", str(exc))
                continue
            except Exception as exc:
                _send_error(self.request, request_id, "internal_error", str(exc))
                continue

            send_message(
                self.request,
                {"id": request_id, "ok": True, "result": result},
                response_payload,
            )


def _send_error(sock, request_id: object, code: str, message: str) -> None:
    send_message(
        sock,
        {
            "id": request_id,
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


def _required_int(metadata: dict[str, Any], key: str) -> int:
    if key not in metadata:
        raise AllocatorError("invalid_request", f"missing {key}")
    value = int(metadata[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


def _optional_list(metadata: dict[str, Any], key: str) -> list[int]:
    value = metadata.get(key, [])
    if not isinstance(value, list):
        raise AllocatorError("invalid_request", f"{key} must be a list")
    return [int(item) for item in value]


def make_runtime(args: argparse.Namespace) -> BonsaiRuntime:
    total_bytes = args.heap_mib * MIB
    slab_bytes = args.slab_mib * MIB
    if args.allocator == "fake":
        allocator = fake_allocator(total_bytes, slab_bytes)
        overlay = None
    elif args.allocator == "pynq":
        overlay = load_overlay(args.overlay)
        allocator = pynq_allocator(total_bytes, slab_bytes)
    else:
        raise ValueError(f"unknown allocator {args.allocator}")
    return BonsaiRuntime(allocator, args.overlay_id, overlay)


def load_overlay(path: str) -> object | None:
    if path == "none":
        return None
    if path == "base":
        from pynq.overlays.base import BaseOverlay

        return BaseOverlay("base.bit")

    from pynq import Overlay

    return Overlay(path)


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
        help="'base', 'none', or a bitfile path for the PYNQ allocator",
    )
    parser.add_argument("--overlay-id", default="fake-local")
    return parser.parse_args(argv)


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
    raise SystemExit(main())
