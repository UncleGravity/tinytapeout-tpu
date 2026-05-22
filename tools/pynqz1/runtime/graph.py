from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

from runtime.allocator import AllocatorError, TensorAllocator


GRAPH_VERSION = 1
Q1_BLOCK = 128
Q1_BLOCK_BYTES = 18
Q8_BLOCK = 32


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

    if op_name == "MATMUL_Q1A8":
        _run_matmul_q1a8(allocator, op, counters)
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


def _run_matmul_q1a8(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
) -> None:
    weights = _required_int(op, "weights")
    acts = _required_int(op, "acts")
    dst = _required_int(op, "dst")
    rows = _required_int(op, "rows")
    cols = _required_int(op, "cols")
    k = _required_int(op, "k")
    weights_offset = _optional_int(op, "weights_offset", 0)
    acts_offset = _optional_int(op, "acts_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    if k == 0 or k % Q1_BLOCK != 0:
        raise AllocatorError(
            "invalid_request",
            f"MATMUL_Q1A8 k must be a positive multiple of {Q1_BLOCK}",
        )

    blocks_per_row = k // Q1_BLOCK
    weight_row_bytes = blocks_per_row * Q1_BLOCK_BYTES
    weight_nbytes = rows * weight_row_bytes
    act_nbytes = cols * k * struct.calcsize("<f")
    dst_nbytes = rows * cols * struct.calcsize("<f")

    weight_data = allocator.read(weights, weights_offset, weight_nbytes)
    act_data = allocator.read(acts, acts_offset, act_nbytes)
    output = bytearray(dst_nbytes)

    for col in range(cols):
        act_row_offset = col * k * struct.calcsize("<f")
        act_row = struct.unpack_from(f"<{k}f", act_data, act_row_offset)
        act_quants, act_scales = _quantize_acts_q8_0(act_row)
        for row in range(rows):
            acc = 0.0
            weight_row_offset = row * weight_row_bytes
            for q1_index in range(blocks_per_row):
                block_offset = weight_row_offset + q1_index * Q1_BLOCK_BYTES
                weight_scale = struct.unpack_from("<e", weight_data, block_offset)[0]
                q1_base = q1_index * Q1_BLOCK
                for q8_base in range(q1_base, q1_base + Q1_BLOCK, Q8_BLOCK):
                    act_scale = act_scales[q8_base // Q8_BLOCK]
                    if weight_scale == 0.0 or act_scale == 0.0:
                        continue
                    sub_sum = _q1_q8_sub_sum(
                        weight_data,
                        block_offset + struct.calcsize("<e"),
                        q1_base,
                        q8_base,
                        act_quants,
                    )
                    acc += weight_scale * act_scale * sub_sum
            struct.pack_into("<f", output, (col * rows + row) * 4, acc)

    allocator.write(dst, dst_offset, output)
    counters.ps_ops += 1
    counters.bytes_read += weight_nbytes + act_nbytes
    counters.bytes_written += dst_nbytes


def _quantize_acts_q8_0(values: tuple[float, ...]) -> tuple[list[int], list[float]]:
    quants = [0] * len(values)
    scales = [0.0] * ((len(values) + Q8_BLOCK - 1) // Q8_BLOCK)

    for block_index, block_start in enumerate(range(0, len(values), Q8_BLOCK)):
        block = values[block_start : block_start + Q8_BLOCK]
        amax = 0.0
        for value in block:
            if math.isfinite(value):
                amax = max(amax, abs(value))
        if amax == 0.0:
            continue

        scale = amax / 127.0
        scales[block_index] = struct.unpack("<e", struct.pack("<e", scale))[0]
        inv_scale = 1.0 / scale
        for local_index, value in enumerate(block):
            if not math.isfinite(value):
                continue
            quant = _lround(value * inv_scale)
            quants[block_start + local_index] = min(127, max(-128, quant))

    return quants, scales


def _q1_q8_sub_sum(
    weights: bytes,
    bits_offset: int,
    q1_base: int,
    q8_base: int,
    act_quants: list[int],
) -> int:
    total = 0
    for index in range(q8_base, q8_base + Q8_BLOCK):
        bit_index = index - q1_base
        bit_byte = weights[bits_offset + bit_index // 8]
        act = act_quants[index]
        total += act if bit_byte & (1 << (bit_index % 8)) else -act
    return total


def _lround(value: float) -> int:
    if value >= 0.0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


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
