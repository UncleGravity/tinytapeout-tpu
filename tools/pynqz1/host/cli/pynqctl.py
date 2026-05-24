from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from host.transport.client import RpcClient, RpcRemoteError  # noqa: E402
from proto.framing import ProtocolError  # noqa: E402
from proto.ops import DEFAULT_PORT  # noqa: E402


class CliError(RuntimeError):
    pass


SIZE_SUFFIXES = {
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024 * 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
    "g": 1024 * 1024 * 1024,
    "gb": 1024 * 1024 * 1024,
    "gib": 1024 * 1024 * 1024,
}


def parse_size(text: str) -> int:
    value = text.strip().lower().replace("_", "")
    split_at = len(value)
    while split_at > 0 and value[split_at - 1].isalpha():
        split_at -= 1

    number = value[:split_at]
    suffix = value[split_at:] or "b"
    if not number:
        raise argparse.ArgumentTypeError(f"invalid byte count: {text}")
    if suffix not in SIZE_SUFFIXES:
        raise argparse.ArgumentTypeError(f"unknown size suffix: {suffix}")

    try:
        parsed = int(number, 0) * SIZE_SUFFIXES[suffix]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte count: {text}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("byte count must be non-negative")
    return parsed


def parse_shape(text: str) -> list[int]:
    if not text:
        return []
    dims = text.replace("x", ",").split(",")
    try:
        shape = [int(dim, 0) for dim in dims if dim]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {text}") from exc
    if any(dim < 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be non-negative")
    return shape


def parse_handle(text: str) -> int:
    try:
        handle = int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid tensor handle: {text}") from exc
    if handle < 0:
        raise argparse.ArgumentTypeError("tensor handle must be non-negative")
    return handle


def print_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def call_runtime(
    args: argparse.Namespace,
    op: str,
    payload: bytes | bytearray | memoryview = b"",
    **fields: Any,
) -> tuple[dict[str, Any], bytes]:
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        client = RpcClient(sock)
        response, response_payload = client.call(op, payload=payload, **fields)
    result = response.get("result", {})
    if not isinstance(result, dict):
        raise ProtocolError("response result must be an object")
    return result, response_payload


def cmd_hello(args: argparse.Namespace) -> int:
    result, _payload = call_runtime(args, "HELLO")
    print_json(result)
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    result, _payload = call_runtime(args, "MEMORY")
    print_json(result)
    return 0


def cmd_alloc(args: argparse.Namespace) -> int:
    result, _payload = call_runtime(
        args,
        "ALLOC_TENSOR",
        nbytes=args.nbytes,
        shape=args.shape,
        dtype=args.dtype,
        usage=args.usage,
        layout=args.layout,
        alignment=args.alignment,
    )
    print_json(result)
    return 0


def cmd_free(args: argparse.Namespace) -> int:
    result, _payload = call_runtime(args, "FREE_TENSOR", handle=args.handle)
    print_json(result)
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    data = read_payload(args.path)
    result, _payload = call_runtime(
        args,
        "UPLOAD_TENSOR",
        payload=data,
        handle=args.handle,
        offset=args.offset,
    )
    result = {**result, "path": args.path, "bytes": len(data)}
    print_json(result)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    result, payload = call_runtime(
        args,
        "DOWNLOAD_TENSOR",
        handle=args.handle,
        offset=args.offset,
        size=args.nbytes,
    )
    write_payload(args.path, payload)
    if args.path == "-":
        return 0

    result = {**result, "path": args.path, "bytes": len(payload)}
    print_json(result)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    source = deterministic_bytes(args.nbytes)
    handle = None
    free_result: dict[str, Any] | None = None

    alloc_result, _payload = call_runtime(
        args,
        "ALLOC_TENSOR",
        nbytes=args.nbytes,
        shape=[args.nbytes],
        dtype="u8",
        usage="smoke",
        layout="raw",
    )
    tensor = alloc_result.get("tensor")
    if not isinstance(tensor, dict):
        raise ProtocolError("ALLOC_TENSOR response did not contain tensor object")
    handle = int(tensor["handle"])

    try:
        upload_result, _payload = call_runtime(
            args,
            "UPLOAD_TENSOR",
            payload=source,
            handle=handle,
            offset=0,
        )
        download_result, payload = call_runtime(
            args,
            "DOWNLOAD_TENSOR",
            handle=handle,
            offset=0,
            size=args.nbytes,
        )
        if payload != source:
            raise CliError("downloaded payload does not match uploaded payload")
    finally:
        if handle is not None:
            free_result, _payload = call_runtime(args, "FREE_TENSOR", handle=handle)

    print_json(
        {
            "ok": True,
            "nbytes": args.nbytes,
            "handle": handle,
            "upload": upload_result,
            "download": download_result,
            "free": free_result,
        }
    )
    return 0


def cmd_graph_copy_smoke(args: argparse.Namespace) -> int:
    source = deterministic_bytes(args.nbytes)
    handles: list[int] = []
    free_results: list[dict[str, Any]] = []

    src_result, _payload = call_runtime(
        args,
        "ALLOC_TENSOR",
        nbytes=args.nbytes,
        shape=[args.nbytes],
        dtype="u8",
        usage="graph-copy-src",
        layout="raw",
    )
    src_handle = tensor_handle(src_result)
    handles.append(src_handle)

    try:
        dst_result, _payload = call_runtime(
            args,
            "ALLOC_TENSOR",
            nbytes=args.nbytes,
            shape=[args.nbytes],
            dtype="u8",
            usage="graph-copy-dst",
            layout="raw",
        )
        dst_handle = tensor_handle(dst_result)
        handles.append(dst_handle)

        upload_result, _payload = call_runtime(
            args,
            "UPLOAD_TENSOR",
            payload=source,
            handle=src_handle,
            offset=0,
        )
        graph_result, _payload = call_runtime(
            args,
            "RUN_GRAPH",
            graph_version=1,
            ops=[
                {
                    "op": "COPY",
                    "src": src_handle,
                    "dst": dst_handle,
                    "nbytes": args.nbytes,
                }
            ],
            outputs=[dst_handle],
        )
        download_result, payload = call_runtime(
            args,
            "DOWNLOAD_TENSOR",
            handle=dst_handle,
            offset=0,
            size=args.nbytes,
        )
        if payload != source:
            raise CliError("graph COPY output does not match uploaded source")
    finally:
        for handle in reversed(handles):
            result, _payload = call_runtime(args, "FREE_TENSOR", handle=handle)
            free_results.append(result)

    print_json(
        {
            "ok": True,
            "nbytes": args.nbytes,
            "source": src_result,
            "destination": dst_result,
            "upload": upload_result,
            "graph": graph_result,
            "download": download_result,
            "free": free_results,
        }
    )
    return 0


def tensor_handle(alloc_result: dict[str, Any]) -> int:
    tensor = alloc_result.get("tensor")
    if not isinstance(tensor, dict):
        raise ProtocolError("ALLOC_TENSOR response did not contain tensor object")
    return int(tensor["handle"])


def read_payload(path: str) -> bytes:
    if path == "-":
        return sys.stdin.buffer.read()
    return Path(path).read_bytes()


def write_payload(path: str, payload: bytes) -> None:
    if path == "-":
        sys.stdout.buffer.write(payload)
        return
    Path(path).write_bytes(payload)


def deterministic_bytes(nbytes: int) -> bytes:
    return bytes((index * 17 + 3) % 251 for index in range(nbytes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control a PYNQ-Z1 daemon")
    parser.add_argument("--host", default=os.environ.get("PYNQ_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PYNQ_PORT", DEFAULT_PORT)))
    parser.add_argument("--timeout", type=float, default=5.0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    hello = subparsers.add_parser("hello", help="query daemon ABI and capabilities")
    hello.set_defaults(func=cmd_hello)

    memory = subparsers.add_parser("memory", help="query board memory state")
    memory.set_defaults(func=cmd_memory)

    alloc = subparsers.add_parser("alloc", help="allocate a tensor on the board")
    alloc.add_argument("--bytes", dest="nbytes", type=parse_size, required=True)
    alloc.add_argument("--shape", type=parse_shape, default=[])
    alloc.add_argument("--dtype", default="u8")
    alloc.add_argument("--usage", default="tensor")
    alloc.add_argument("--layout", default="raw")
    alloc.add_argument("--alignment", type=parse_size, default=64)
    alloc.set_defaults(func=cmd_alloc)

    upload = subparsers.add_parser("upload", help="upload a file into a tensor")
    upload.add_argument("handle", type=parse_handle)
    upload.add_argument("path")
    upload.add_argument("--offset", type=parse_size, default=0)
    upload.set_defaults(func=cmd_upload)

    download = subparsers.add_parser("download", help="download tensor bytes to a file")
    download.add_argument("handle", type=parse_handle)
    download.add_argument("path")
    download.add_argument("--bytes", dest="nbytes", type=parse_size, required=True)
    download.add_argument("--offset", type=parse_size, default=0)
    download.set_defaults(func=cmd_download)

    free = subparsers.add_parser("free", help="free a tensor")
    free.add_argument("handle", type=parse_handle)
    free.set_defaults(func=cmd_free)

    smoke = subparsers.add_parser("smoke", help="allocate/upload/download/free check")
    smoke.add_argument("--bytes", dest="nbytes", type=parse_size, default=64 * 1024)
    smoke.set_defaults(func=cmd_smoke)

    graph_copy_smoke = subparsers.add_parser(
        "graph-copy-smoke",
        help="verify RUN_GRAPH COPY with board-resident source and destination tensors",
    )
    graph_copy_smoke.add_argument(
        "--bytes", dest="nbytes", type=parse_size, default=64 * 1024
    )
    graph_copy_smoke.set_defaults(func=cmd_graph_copy_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RpcRemoteError as exc:
        print(f"remote error: {exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except (CliError, OSError, ProtocolError) as exc:
        print(f"pynqctl: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
