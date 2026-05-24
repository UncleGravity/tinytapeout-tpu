"""``RUN_GRAPH`` handler.

The dispatch loop is a flat table-driven walk: parse op metadata, look
up the matching kernel in the registry, run it under a ``Timer``. There
are no per-op branches, no per-op handler functions, and no Python
fallbacks. Adding a new op = one ``registry.register`` call.
"""

from __future__ import annotations

from typing import Any

from board.kernels.registry import KernelRegistry
from board.memory.allocator import AllocatorError, TensorAllocator
from board.profiling.timer import Timer
from proto.ops import (
    F_COUNTERS,
    F_GRAPH_VERSION,
    F_OP,
    F_OPS,
    F_OP_COUNT,
    F_OUTPUTS,
    GRAPH_VERSION,
)


def run_graph(
    allocator: TensorAllocator,
    registry: KernelRegistry,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    graph_version = _required_int(metadata, F_GRAPH_VERSION)
    if graph_version != GRAPH_VERSION:
        raise AllocatorError(
            "unsupported_graph_version",
            f"unsupported graph_version {graph_version}",
        )

    ops = _required_ops(metadata)
    outputs = _optional_int_list(metadata, F_OUTPUTS)

    timer = Timer()
    with timer.section("graph"):
        for index, op in enumerate(ops):
            op_name = str(op.get(F_OP, ""))
            if not op_name:
                raise AllocatorError(
                    "invalid_request", f"graph op {index} is missing op"
                )
            kernel = registry.get(op_name)
            with timer.op(op_name, index=index):
                kernel.run(allocator, op, timer)

    # Validate every declared output exists.
    for handle in outputs:
        allocator.describe(handle)

    counters = {
        "ps_ops": len(ops),
        "pl_ops": 0,
        "bytes_read": sum(int(s.fields.get("bytes_read", 0)) for s in timer.ops),
        "bytes_written": sum(int(s.fields.get("bytes_written", 0)) for s in timer.ops),
        "elapsed_us": timer.graph_us,
    }
    timer.emit_if_enabled(counters=counters)

    return {
        F_GRAPH_VERSION: graph_version,
        F_OP_COUNT: len(ops),
        F_OUTPUTS: outputs,
        F_COUNTERS: counters,
    }


# -- metadata helpers ------------------------------------------------------


def _required_ops(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    ops = metadata.get(F_OPS)
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


def _non_negative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise AllocatorError("invalid_request", f"{name} must be non-negative")
    return parsed
