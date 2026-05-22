from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.allocator import AllocatorError, TensorAllocator


GRAPH_VERSION = 1


@dataclass
class GraphCounters:
    ps_ops: int = 0
    pl_ops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

    def describe(self) -> dict[str, int]:
        return {
            "ps_ops": self.ps_ops,
            "pl_ops": self.pl_ops,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }


def run_graph(
    allocator: TensorAllocator, metadata: dict[str, Any]
) -> dict[str, object]:
    graph_version = _required_int(metadata, "graph_version")
    if graph_version != GRAPH_VERSION:
        raise AllocatorError(
            "unsupported_graph_version",
            f"unsupported graph_version {graph_version}",
        )

    ops = _required_ops(metadata)
    outputs = _optional_int_list(metadata, "outputs")
    counters = GraphCounters()

    for index, op in enumerate(ops):
        _run_op(allocator, op, index, counters)

    for handle in outputs:
        allocator.describe(handle)

    return {
        "graph_version": graph_version,
        "op_count": len(ops),
        "outputs": outputs,
        "counters": counters.describe(),
    }


def _run_op(
    allocator: TensorAllocator,
    op: dict[str, Any],
    index: int,
    counters: GraphCounters,
) -> None:
    op_name = str(op.get("op", ""))
    if not op_name:
        raise AllocatorError("invalid_request", f"graph op {index} is missing op")

    if op_name == "COPY":
        _run_copy(allocator, op, counters)
        return

    raise AllocatorError(
        "unsupported_op",
        f"unsupported graph op {op_name} at index {index}",
    )


def _run_copy(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
) -> None:
    src = _required_int(op, "src")
    dst = _required_int(op, "dst")
    nbytes = _required_int(op, "nbytes")
    src_offset = _optional_int(op, "src_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    data = allocator.read(src, src_offset, nbytes)
    allocator.write(dst, dst_offset, data)
    counters.ps_ops += 1
    counters.bytes_read += nbytes
    counters.bytes_written += nbytes


def _required_ops(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    ops = metadata.get("ops")
    if not isinstance(ops, list):
        raise AllocatorError("invalid_request", "ops must be a list")
    if not all(isinstance(op, dict) for op in ops):
        raise AllocatorError("invalid_request", "ops must contain objects")
    return ops


def _optional_int_list(metadata: dict[str, Any], key: str) -> list[int]:
    value = metadata.get(key, [])
    if not isinstance(value, list):
        raise AllocatorError("invalid_request", f"{key} must be a list")
    return [_non_negative_int(item, key) for item in value]


def _required_int(metadata: dict[str, Any], key: str) -> int:
    if key not in metadata:
        raise AllocatorError("invalid_request", f"missing {key}")
    return _non_negative_int(metadata[key], key)


def _optional_int(metadata: dict[str, Any], key: str, default: int) -> int:
    if key not in metadata:
        return default
    return _non_negative_int(metadata[key], key)


def _non_negative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise AllocatorError("invalid_request", f"{name} must be non-negative")
    return parsed
