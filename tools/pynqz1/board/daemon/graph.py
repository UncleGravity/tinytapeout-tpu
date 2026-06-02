"""``RUN_GRAPH`` handler.

The dispatch loop is a single-in-flight pipelining scheduler. Ops run in
order, but a kernel that exposes ``run_async`` (today only the PL matmul)
may issue its DMA and return a *pending* handle; subsequent ops that don't
touch the in-flight result run on the ARM while the PL streams, and the
pending op is ``complete``d the moment an op needs its output (or another
matmul needs the kernel). Kernels without ``run_async`` run synchronously,
so the loop is behaviourally identical to the old flat walk for them.

Dependency safety rests on ggml's compute-arena guarantee: two
simultaneously-live tensors never overlap in memory. So the only op that
touches the in-flight matmul's destination bytes is one that reads (or
overwrites) that tensor — detectable by its tensor handle+offset landing
inside the matmul's dst byte interval.
"""

from __future__ import annotations

import time
from typing import Any

from board.kernels.registry import KernelRegistry
from board.memory.allocator import AllocatorError, TensorAllocator
from board.profiling import events
from board.profiling.timer import Timer
from proto.ops import (
    F_ACTS,
    F_ACTS_OFFSET,
    F_COUNTERS,
    F_DST,
    F_DST_OFFSET,
    F_GRAPH_VERSION,
    F_ID,
    F_INDICES,
    F_INDICES_OFFSET,
    F_K_OFFSET,
    F_K_TENSOR,
    F_MASK,
    F_MASK_OFFSET,
    F_OP,
    F_OP_COUNT,
    F_OPS,
    F_OUTPUTS,
    F_POSITIONS,
    F_POSITIONS_OFFSET,
    F_SRC,
    F_SRC0,
    F_SRC0_OFFSET,
    F_SRC1,
    F_SRC1_OFFSET,
    F_SRC_OFFSET,
    F_V_OFFSET,
    F_V_TENSOR,
    F_WEIGHTS,
    F_WEIGHTS_OFFSET,
    GRAPH_VERSION,
)

# (handle field, offset field) for every tensor an op can reference. Used to
# test whether an op touches the in-flight matmul's result range.
_TENSOR_ROLES: tuple[tuple[str, str], ...] = (
    (F_SRC, F_SRC_OFFSET),
    (F_DST, F_DST_OFFSET),
    (F_SRC0, F_SRC0_OFFSET),
    (F_SRC1, F_SRC1_OFFSET),
    (F_WEIGHTS, F_WEIGHTS_OFFSET),
    (F_ACTS, F_ACTS_OFFSET),
    (F_POSITIONS, F_POSITIONS_OFFSET),
    (F_K_TENSOR, F_K_OFFSET),
    (F_V_TENSOR, F_V_OFFSET),
    (F_MASK, F_MASK_OFFSET),
    (F_INDICES, F_INDICES_OFFSET),
)


def _op_touches(op: dict[str, Any], handle: int, lo: int, hi: int) -> bool:
    """True if ``op`` references byte ``[lo, hi)`` of tensor ``handle``.

    Sound under ggml's arena non-overlap guarantee: an op that reads or
    overwrites the in-flight matmul's dst tensor references it at a (handle,
    offset) whose offset lands in the dst byte interval; no *other* live
    tensor can occupy those bytes. Unknown offsets default to 0, which is
    conservative (more likely to land in [lo, hi) and force a wait)."""
    for handle_field, offset_field in _TENSOR_ROLES:
        if handle_field in op and int(op[handle_field]) == handle:
            offset = int(op.get(offset_field, 0))
            if lo <= offset < hi:
                return True
    return False


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
    req_id = metadata.get(F_ID)

    timer = Timer(req_id=req_id)
    counts = {"ps": 0, "pl": 0}
    events.emit("graph_begin", req_id=req_id, op_count=len(ops))
    # Elapsed is measured directly (not via the timer) so the response counter
    # is correct whether or not per-op profiling is enabled.
    graph_start_ns = time.perf_counter_ns()
    with timer.section("graph"):
        # The single in-flight pending op (a matmul whose DMA is streaming),
        # or None. `.pending` is opaque to the scheduler; `.dst_*` bound the
        # bytes downstream ops must not touch until it completes.
        in_flight = None

        def complete(pending) -> None:
            # Attribute the wait + result handling to a span named like the op
            # so the profile aggregates it with the issue span.
            with timer.op(pending.op_name):
                pending.kernel.complete(pending, timer)

        for index, op in enumerate(ops):
            op_name = str(op.get(F_OP, ""))
            if not op_name:
                raise AllocatorError(
                    "invalid_request", f"graph op {index} is missing op"
                )
            kernel = registry.get(op_name)

            # Barrier: finish the in-flight matmul before an op that consumes
            # its result, or before another op that needs the PL kernel.
            if in_flight is not None and (
                getattr(kernel, "backend", "ps") == "pl"
                or _op_touches(op, in_flight.dst_handle,
                               in_flight.dst_lo, in_flight.dst_hi)
            ):
                complete(in_flight)
                in_flight = None

            run_async = getattr(kernel, "run_async", None)
            if run_async is not None and in_flight is None:
                with timer.op(op_name):
                    pending = run_async(allocator, op, timer)
                counts["pl" if getattr(kernel, "backend", "ps") == "pl"
                       else "ps"] += 1
                if pending is not None:
                    pending.op_name = op_name
                    pending.kernel = kernel
                    in_flight = pending
                continue

            counts["pl" if getattr(kernel, "backend", "ps") == "pl"
                   else "ps"] += 1
            with timer.op(op_name):
                kernel.run(allocator, op, timer)

        if in_flight is not None:
            complete(in_flight)
            in_flight = None
    elapsed_us = max(0, (time.perf_counter_ns() - graph_start_ns) // 1000)
    ps_ops = counts["ps"]
    pl_ops = counts["pl"]

    # Validate every declared output exists.
    for handle in outputs:
        allocator.describe(handle)

    counters = {
        "ps_ops": ps_ops,
        "pl_ops": pl_ops,
        "bytes_read": timer.total_bytes_read,
        "bytes_written": timer.total_bytes_written,
        "elapsed_us": elapsed_us,
    }
    events.emit("graph_end", req_id=req_id, **counters)

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
