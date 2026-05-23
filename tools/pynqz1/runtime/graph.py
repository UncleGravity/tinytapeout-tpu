from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

from runtime.allocator import AllocatorError, TensorAllocator
from runtime.ps_native import get_native_kernels


GRAPH_VERSION = 1
Q1_BLOCK = 128
Q1_BLOCK_BYTES = 18
Q8_BLOCK = 32
F32_BYTES = struct.calcsize("<f")


@dataclass
class GraphCounters:
    ps_ops: int = 0
    pl_ops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    elapsed_us: int = 0

    def describe(self) -> dict[str, int]:
        return {
            "ps_ops": self.ps_ops,
            "pl_ops": self.pl_ops,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "elapsed_us": self.elapsed_us,
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
    profile_ops: list[dict[str, Any]] | None = [] if _profile_enabled() else None
    start_ns = time.perf_counter_ns()

    for index, op in enumerate(ops):
        op_profile = _new_op_profile(op, index) if profile_ops is not None else None
        op_start_ns = time.perf_counter_ns() if op_profile is not None else 0
        _run_op(allocator, op, index, counters, op_profile)
        if op_profile is not None:
            op_profile["total_us"] = _elapsed_us(op_start_ns)
            profile_ops.append(op_profile)
    counters.elapsed_us = _elapsed_us(start_ns)

    for handle in outputs:
        allocator.describe(handle)

    if profile_ops is not None:
        _emit_profile(counters.elapsed_us, profile_ops, counters)

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
    profile: dict[str, Any] | None,
) -> None:
    op_name = str(op.get("op", ""))
    if not op_name:
        raise AllocatorError("invalid_request", f"graph op {index} is missing op")

    if op_name == "COPY":
        _run_copy(allocator, op, counters, profile)
        return

    if op_name == "MATMUL_Q1A8":
        _run_matmul_q1a8(allocator, op, counters, profile)
        return

    if op_name == "ADD_F32":
        _run_binary_f32(
            allocator,
            op,
            counters,
            profile,
            "add",
            lambda lhs, rhs: lhs + rhs,
        )
        return

    if op_name == "MUL_F32":
        _run_binary_f32(
            allocator,
            op,
            counters,
            profile,
            "mul",
            lambda lhs, rhs: lhs * rhs,
        )
        return

    if op_name == "SCALE_F32":
        _run_scale_f32(allocator, op, counters, profile)
        return

    if op_name == "SILU_F32":
        _run_silu_f32(allocator, op, counters, profile)
        return

    if op_name == "SWIGLU_F32":
        _run_swiglu_f32(allocator, op, counters, profile)
        return

    if op_name == "RMS_NORM_F32":
        _run_rms_norm_f32(allocator, op, counters, profile)
        return

    raise AllocatorError(
        "unsupported_op",
        f"unsupported graph op {op_name} at index {index}",
    )


def _run_copy(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
) -> None:
    src = _required_int(op, "src")
    dst = _required_int(op, "dst")
    nbytes = _required_int(op, "nbytes")
    src_offset = _optional_int(op, "src_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    if profile is None:
        data = allocator.read(src, src_offset, nbytes)
        allocator.write(dst, dst_offset, data)
    else:
        read_start_ns = time.perf_counter_ns()
        data = allocator.read(src, src_offset, nbytes)
        _add_elapsed_us(profile, "read_us", read_start_ns)

        write_start_ns = time.perf_counter_ns()
        allocator.write(dst, dst_offset, data)
        _add_elapsed_us(profile, "write_us", write_start_ns)
        profile["bytes_read"] = nbytes
        profile["bytes_written"] = nbytes

    counters.ps_ops += 1
    counters.bytes_read += nbytes
    counters.bytes_written += nbytes


def _run_matmul_q1a8(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
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
    act_nbytes = cols * k * F32_BYTES
    dst_nbytes = rows * cols * F32_BYTES

    if profile is None:
        weight_data = allocator.read(weights, weights_offset, weight_nbytes)
        act_data = allocator.read(acts, acts_offset, act_nbytes)
    else:
        weight_read_start_ns = time.perf_counter_ns()
        weight_data = allocator.read(weights, weights_offset, weight_nbytes)
        weight_read_us = _elapsed_us(weight_read_start_ns)
        profile["weight_read_us"] = weight_read_us
        profile["read_us"] += weight_read_us

        act_read_start_ns = time.perf_counter_ns()
        act_data = allocator.read(acts, acts_offset, act_nbytes)
        act_read_us = _elapsed_us(act_read_start_ns)
        profile["act_read_us"] = act_read_us
        profile["read_us"] += act_read_us
        profile["bytes_read"] = weight_nbytes + act_nbytes
        profile["bytes_written"] = dst_nbytes

    native = get_native_kernels()
    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(dst_nbytes)
    native_used = native.matmul_q1a8(
        weight_data,
        act_data,
        output,
        rows,
        cols,
        k,
        profile,
    )
    if not native_used:
        _run_matmul_q1a8_python(
            weight_data,
            act_data,
            output,
            rows,
            cols,
            k,
            blocks_per_row,
            weight_row_bytes,
        )
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used

    if profile is None:
        allocator.write(dst, dst_offset, output)
    else:
        write_start_ns = time.perf_counter_ns()
        allocator.write(dst, dst_offset, output)
        _add_elapsed_us(profile, "write_us", write_start_ns)

    counters.ps_ops += 1
    counters.bytes_read += weight_nbytes + act_nbytes
    counters.bytes_written += dst_nbytes


def _run_matmul_q1a8_python(
    weight_data: bytes,
    act_data: bytes,
    output: bytearray,
    rows: int,
    cols: int,
    k: int,
    blocks_per_row: int,
    weight_row_bytes: int,
) -> None:
    for col in range(cols):
        act_row_offset = col * k * F32_BYTES
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
            struct.pack_into("<f", output, (col * rows + row) * F32_BYTES, acc)


def _run_binary_f32(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
    native_op: str,
    fn,
) -> None:
    src0 = _required_int(op, "src0")
    src1 = _required_int(op, "src1")
    dst = _required_int(op, "dst")
    rows = _positive_int(op, "rows")
    cols = _positive_int(op, "cols")
    src0_offset = _optional_int(op, "src0_offset", 0)
    src1_offset = _optional_int(op, "src1_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)
    src1_broadcast = _optional_bool(op, "src1_broadcast", False)

    elements = rows * cols
    rhs_elements = rows if src1_broadcast else elements
    src0_data = _read_bytes(allocator, src0, src0_offset, elements * F32_BYTES, profile)
    src1_data = _read_bytes(
        allocator,
        src1,
        src1_offset,
        rhs_elements * F32_BYTES,
        profile,
    )

    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(elements * F32_BYTES)
    native = get_native_kernels()
    native_used = native.binary_f32(
        native_op,
        src0_data,
        src1_data,
        output,
        rows,
        cols,
        src1_broadcast,
        profile,
    )
    if not native_used:
        src0_values = _unpack_f32(src0_data, elements)
        src1_values = _unpack_f32(src1_data, rhs_elements)
        output_values = [0.0] * elements
        for col in range(cols):
            col_offset = col * rows
            for row in range(rows):
                index = col_offset + row
                rhs_index = row if src1_broadcast else index
                output_values[index] = fn(src0_values[index], src1_values[rhs_index])
        output = bytearray(struct.pack(f"<{elements}f", *output_values))
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used

    _write_bytes(allocator, dst, dst_offset, output, profile)
    if profile is not None:
        profile["bytes_read"] = (elements + rhs_elements) * F32_BYTES
        profile["bytes_written"] = elements * F32_BYTES
    counters.ps_ops += 1
    counters.bytes_read += (elements + rhs_elements) * F32_BYTES
    counters.bytes_written += elements * F32_BYTES


def _run_scale_f32(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
) -> None:
    src = _required_int(op, "src")
    dst = _required_int(op, "dst")
    elements = _positive_int(op, "elements")
    scale = _required_float(op, "scale")
    bias = _optional_float(op, "bias", 0.0)
    src_offset = _optional_int(op, "src_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    data = _read_bytes(allocator, src, src_offset, elements * F32_BYTES, profile)
    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(elements * F32_BYTES)
    native = get_native_kernels()
    native_used = native.scale_f32(data, output, elements, scale, bias, profile)
    if not native_used:
        values = _unpack_f32(data, elements)
        output = bytearray(
            struct.pack(f"<{elements}f", *((value * scale) + bias for value in values))
        )
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used
    _write_bytes(allocator, dst, dst_offset, output, profile)
    if profile is not None:
        profile["bytes_read"] = elements * F32_BYTES
        profile["bytes_written"] = elements * F32_BYTES
    counters.ps_ops += 1
    counters.bytes_read += elements * F32_BYTES
    counters.bytes_written += elements * F32_BYTES


def _run_silu_f32(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
) -> None:
    src = _required_int(op, "src")
    dst = _required_int(op, "dst")
    elements = _positive_int(op, "elements")
    src_offset = _optional_int(op, "src_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    data = _read_bytes(allocator, src, src_offset, elements * F32_BYTES, profile)
    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(elements * F32_BYTES)
    native = get_native_kernels()
    native_used = native.silu_f32(data, output, elements, profile)
    if not native_used:
        values = _unpack_f32(data, elements)
        output = bytearray(struct.pack(f"<{elements}f", *(_silu(value) for value in values)))
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used
    _write_bytes(allocator, dst, dst_offset, output, profile)
    if profile is not None:
        profile["bytes_read"] = elements * F32_BYTES
        profile["bytes_written"] = elements * F32_BYTES
    counters.ps_ops += 1
    counters.bytes_read += elements * F32_BYTES
    counters.bytes_written += elements * F32_BYTES


def _run_swiglu_f32(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
) -> None:
    src0 = _required_int(op, "src0")
    src1 = _required_int(op, "src1")
    dst = _required_int(op, "dst")
    elements = _positive_int(op, "elements")
    src0_offset = _optional_int(op, "src0_offset", 0)
    src1_offset = _optional_int(op, "src1_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    gate_data = _read_bytes(allocator, src0, src0_offset, elements * F32_BYTES, profile)
    up_data = _read_bytes(allocator, src1, src1_offset, elements * F32_BYTES, profile)
    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(elements * F32_BYTES)
    native = get_native_kernels()
    native_used = native.swiglu_f32(gate_data, up_data, output, elements, profile)
    if not native_used:
        gate_values = _unpack_f32(gate_data, elements)
        up_values = _unpack_f32(up_data, elements)
        output = bytearray(
            struct.pack(
                f"<{elements}f",
                *(_silu(gate) * up for gate, up in zip(gate_values, up_values)),
            )
        )
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used
    _write_bytes(allocator, dst, dst_offset, output, profile)
    if profile is not None:
        profile["bytes_read"] = 2 * elements * F32_BYTES
        profile["bytes_written"] = elements * F32_BYTES
    counters.ps_ops += 1
    counters.bytes_read += 2 * elements * F32_BYTES
    counters.bytes_written += elements * F32_BYTES


def _run_rms_norm_f32(
    allocator: TensorAllocator,
    op: dict[str, Any],
    counters: GraphCounters,
    profile: dict[str, Any] | None,
) -> None:
    src = _required_int(op, "src")
    dst = _required_int(op, "dst")
    rows = _positive_int(op, "rows")
    cols = _positive_int(op, "cols")
    eps = _required_float(op, "eps")
    src_offset = _optional_int(op, "src_offset", 0)
    dst_offset = _optional_int(op, "dst_offset", 0)

    elements = rows * cols
    data = _read_bytes(allocator, src, src_offset, elements * F32_BYTES, profile)
    compute_start_ns = time.perf_counter_ns() if profile is not None else 0
    output = bytearray(elements * F32_BYTES)
    native = get_native_kernels()
    native_used = native.rms_norm_f32(data, output, rows, cols, eps, profile)
    if not native_used:
        values = _unpack_f32(data, elements)
        output_values = [0.0] * elements
        for col in range(cols):
            col_offset = col * rows
            row_values = values[col_offset : col_offset + rows]
            mean_square = sum(value * value for value in row_values) / rows
            scale = 1.0 / math.sqrt(mean_square + eps)
            for row, value in enumerate(row_values):
                output_values[col_offset + row] = value * scale
        output = bytearray(struct.pack(f"<{elements}f", *output_values))
    if profile is not None:
        _add_elapsed_us(profile, "compute_us", compute_start_ns)
        profile["native"] = native_used

    _write_bytes(allocator, dst, dst_offset, output, profile)
    if profile is not None:
        profile["bytes_read"] = elements * F32_BYTES
        profile["bytes_written"] = elements * F32_BYTES
    counters.ps_ops += 1
    counters.bytes_read += elements * F32_BYTES
    counters.bytes_written += elements * F32_BYTES


def _read_f32(
    allocator: TensorAllocator,
    handle: int,
    offset: int,
    elements: int,
    profile: dict[str, Any] | None = None,
) -> tuple[float, ...]:
    data = _read_bytes(allocator, handle, offset, elements * F32_BYTES, profile)
    return _unpack_f32(data, elements)


def _write_f32(
    allocator: TensorAllocator,
    handle: int,
    offset: int,
    values: list[float],
    profile: dict[str, Any] | None = None,
) -> None:
    _write_bytes(
        allocator,
        handle,
        offset,
        struct.pack(f"<{len(values)}f", *values),
        profile,
    )


def _read_bytes(
    allocator: TensorAllocator,
    handle: int,
    offset: int,
    nbytes: int,
    profile: dict[str, Any] | None = None,
) -> bytes:
    start_ns = time.perf_counter_ns() if profile is not None else 0
    data = allocator.read(handle, offset, nbytes)
    if profile is not None:
        _add_elapsed_us(profile, "read_us", start_ns)
    return data


def _write_bytes(
    allocator: TensorAllocator,
    handle: int,
    offset: int,
    data: bytes | bytearray,
    profile: dict[str, Any] | None = None,
) -> None:
    start_ns = time.perf_counter_ns() if profile is not None else 0
    allocator.write(handle, offset, data)
    if profile is not None:
        _add_elapsed_us(profile, "write_us", start_ns)


def _unpack_f32(data: bytes, elements: int) -> tuple[float, ...]:
    return struct.unpack(f"<{elements}f", data)


def _profile_enabled() -> bool:
    value = os.environ.get("PYNQ_PROFILE")
    return value is not None and value.lower() not in ("", "0", "false", "no", "off")


def _new_op_profile(op: dict[str, Any], index: int) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "index": index,
        "op": str(op.get("op", "")),
        "read_us": 0,
        "compute_us": 0,
        "write_us": 0,
        "total_us": 0,
        "bytes_read": 0,
        "bytes_written": 0,
    }
    name = op.get("name")
    if isinstance(name, str) and name:
        profile["name"] = name
    for key in ("rows", "cols", "k", "elements", "nbytes"):
        if key in op:
            profile[key] = int(op[key])
    return profile


def _emit_profile(
    graph_us: int,
    ops: list[dict[str, Any]],
    counters: GraphCounters,
) -> None:
    payload = {
        "graph_us": graph_us,
        "counters": counters.describe(),
        "ops": ops,
    }
    print(
        "pynq profile: " + json.dumps(payload, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


def _elapsed_us(start_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - start_ns) // 1000)


def _add_elapsed_us(profile: dict[str, Any], key: str, start_ns: int) -> None:
    profile[key] = int(profile.get(key, 0)) + _elapsed_us(start_ns)


def _silu(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


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


def _positive_int(metadata: dict[str, Any], key: str) -> int:
    parsed = _required_int(metadata, key)
    if parsed <= 0:
        raise AllocatorError("invalid_request", f"{key} must be positive")
    return parsed


def _required_float(metadata: dict[str, Any], key: str) -> float:
    if key not in metadata:
        raise AllocatorError("invalid_request", f"missing {key}")
    return float(metadata[key])


def _optional_float(metadata: dict[str, Any], key: str, default: float) -> float:
    if key not in metadata:
        return default
    return float(metadata[key])


def _optional_bool(metadata: dict[str, Any], key: str, default: bool) -> bool:
    if key not in metadata:
        return default
    value = metadata[key]
    if not isinstance(value, bool):
        raise AllocatorError("invalid_request", f"{key} must be boolean")
    return value


def _non_negative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise AllocatorError("invalid_request", f"{name} must be non-negative")
    return parsed
