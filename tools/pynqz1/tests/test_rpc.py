from __future__ import annotations

import json
import math
import os
import socket
import struct
import threading

import pytest

from runtime.allocator import fake_allocator
from runtime.bonsai_rpc import RpcClient, RpcRemoteError
from runtime.bonsaid import BonsaiRpcServer, BonsaiRuntime
from runtime.graph import run_graph
from runtime.ps_native import get_native_kernels


@pytest.fixture
def rpc_server():
    allocator = fake_allocator(total_bytes=1024 * 1024, slab_bytes=128 * 1024)
    runtime = BonsaiRuntime(allocator, overlay_id="test-overlay")
    server = BonsaiRpcServer(("127.0.0.1", 0), runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def client(rpc_server):
    with socket.create_connection(rpc_server.server_address, timeout=2) as sock:
        yield RpcClient(sock)


def test_hello_reports_memory_and_capabilities(client):
    response, payload = client.call("HELLO")

    assert payload == b""
    result = response["result"]
    assert result["abi_version"] == 1
    assert result["server"] == "bonsaid"
    assert result["overlay_id"] == "test-overlay"
    assert result["memory"]["total_bytes"] == 1024 * 1024
    assert "ALLOC_TENSOR" in result["capabilities"]
    assert "DOWNLOAD_TENSOR" in result["capabilities"]
    assert "RUN_GRAPH" in result["capabilities"]
    assert result["graph_ops"] == [
        "COPY",
        "MATMUL_Q1A8",
        "ADD_F32",
        "MUL_F32",
        "SCALE_F32",
        "SILU_F32",
        "SWIGLU_F32",
        "RMS_NORM_F32",
    ]


def test_native_kernel_loads_when_configured():
    if "PYNQ_PS_LIB" not in os.environ:
        pytest.skip("PYNQ_PS_LIB is not configured")

    native = get_native_kernels()
    assert native.available


def assert_counters(result, expected):
    counters = dict(result["counters"])
    elapsed_us = counters.pop("elapsed_us")
    assert elapsed_us >= 0
    assert counters == expected


def test_allocate_upload_download_and_free_cross_slab_tensor(client):
    nbytes = 300 * 1024
    source = bytes((i * 17 + 3) % 251 for i in range(nbytes))

    response, _ = client.call(
        "ALLOC_TENSOR",
        nbytes=nbytes,
        shape=[1024, 300],
        dtype="u8",
        usage="activation",
        layout="raw",
    )
    tensor = response["result"]["tensor"]
    handle = tensor["handle"]

    assert tensor["nbytes"] == nbytes
    assert tensor["shape"] == [1024, 300]
    assert tensor["extent_count"] >= 3

    response, payload = client.call(
        "UPLOAD_TENSOR",
        payload=source,
        handle=handle,
        offset=0,
    )
    assert payload == b""
    assert response["result"]["written"] == nbytes

    response, payload = client.call(
        "DOWNLOAD_TENSOR",
        handle=handle,
        offset=65531,
        size=100_000,
    )
    assert response["result"]["read"] == 100_000
    assert payload == source[65531 : 65531 + 100_000]

    response, payload = client.call("FREE_TENSOR", handle=handle)
    assert payload == b""
    memory = response["result"]["memory"]
    assert memory["free_bytes"] == memory["total_bytes"]
    assert memory["tensor_count"] == 0


def test_bounds_errors_are_reported(client):
    response, _ = client.call("ALLOC_TENSOR", nbytes=16)
    handle = response["result"]["tensor"]["handle"]

    with pytest.raises(RpcRemoteError) as exc:
        client.call("DOWNLOAD_TENSOR", handle=handle, offset=8, size=16)

    assert exc.value.code == "out_of_bounds"


def test_run_graph_copy_keeps_output_on_device_until_download(client):
    nbytes = 300 * 1024
    source = bytes((i * 31 + 7) % 251 for i in range(nbytes))

    response, _ = client.call("ALLOC_TENSOR", nbytes=nbytes, usage="copy-src")
    src = response["result"]["tensor"]["handle"]
    response, _ = client.call("ALLOC_TENSOR", nbytes=nbytes, usage="copy-dst")
    dst = response["result"]["tensor"]["handle"]
    client.call("UPLOAD_TENSOR", payload=source, handle=src)

    response, payload = client.call(
        "RUN_GRAPH",
        graph_version=1,
        ops=[{"op": "COPY", "src": src, "dst": dst, "nbytes": nbytes}],
        outputs=[dst],
    )

    assert payload == b""
    result = response["result"]
    assert result["graph_version"] == 1
    assert result["op_count"] == 1
    assert result["outputs"] == [dst]
    assert_counters(result, {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": nbytes,
        "bytes_written": nbytes,
    })

    response, payload = client.call("DOWNLOAD_TENSOR", handle=dst, size=nbytes)
    assert response["result"]["read"] == nbytes
    assert payload == source


def test_run_graph_matmul_q1a8_writes_f32_output(client):
    rows = 2
    cols = 2
    k = 128
    block_bytes = 18

    row0_bits = bytes([0xFF] * 16)
    row1_bits = bytes([0x55] * 16)
    weights = struct.pack("<e", 1.0) + row0_bits
    weights += struct.pack("<e", 1.0) + row1_bits
    acts = struct.pack(f"<{cols * k}f", *([0.0] * k + [1.0] * k))

    response, _ = client.call(
        "ALLOC_TENSOR",
        nbytes=rows * block_bytes,
        dtype="Q1_0",
        usage="weights",
    )
    weights_handle = response["result"]["tensor"]["handle"]
    response, _ = client.call(
        "ALLOC_TENSOR",
        nbytes=len(acts),
        dtype="F32",
        usage="activation",
    )
    acts_handle = response["result"]["tensor"]["handle"]
    response, _ = client.call(
        "ALLOC_TENSOR",
        nbytes=rows * cols * 4,
        dtype="F32",
        usage="output",
    )
    dst_handle = response["result"]["tensor"]["handle"]
    client.call("UPLOAD_TENSOR", payload=weights, handle=weights_handle)
    client.call("UPLOAD_TENSOR", payload=acts, handle=acts_handle)

    response, payload = client.call(
        "RUN_GRAPH",
        graph_version=1,
        ops=[
            {
                "op": "MATMUL_Q1A8",
                "weights": weights_handle,
                "acts": acts_handle,
                "dst": dst_handle,
                "rows": rows,
                "cols": cols,
                "k": k,
            }
        ],
        outputs=[dst_handle],
    )

    assert payload == b""
    assert_counters(response["result"], {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": len(weights) + len(acts),
        "bytes_written": rows * cols * 4,
    })

    _, payload = client.call("DOWNLOAD_TENSOR", handle=dst_handle, size=16)
    got = struct.unpack("<4f", payload)
    q8_scale = struct.unpack("<e", struct.pack("<e", 1.0 / 127.0))[0]
    assert got == pytest.approx((0.0, 0.0, 128.0 * 127.0 * q8_scale, 0.0))


def test_run_graph_f32_glue_ops(client):
    rows = 4
    cols = 2
    values = [0.5, -1.0, 2.0, -4.0, 1.5, -2.0, 3.0, -6.0]
    bias = [1.0, 0.5, -0.25, 2.0]
    nbytes = len(values) * 4

    response, _ = client.call("ALLOC_TENSOR", nbytes=nbytes, dtype="F32")
    src = response["result"]["tensor"]["handle"]
    response, _ = client.call("ALLOC_TENSOR", nbytes=len(bias) * 4, dtype="F32")
    row_bias = response["result"]["tensor"]["handle"]
    handles = []
    for _ in range(5):
        response, _ = client.call("ALLOC_TENSOR", nbytes=nbytes, dtype="F32")
        handles.append(response["result"]["tensor"]["handle"])

    client.call("UPLOAD_TENSOR", payload=struct.pack("<8f", *values), handle=src)
    client.call("UPLOAD_TENSOR", payload=struct.pack("<4f", *bias), handle=row_bias)

    add_out, scale_out, norm_out, silu_out, mul_out = handles
    response, payload = client.call(
        "RUN_GRAPH",
        graph_version=1,
        ops=[
            {
                "op": "ADD_F32",
                "src0": src,
                "src1": row_bias,
                "dst": add_out,
                "rows": rows,
                "cols": cols,
                "src1_broadcast": True,
            },
            {
                "op": "SCALE_F32",
                "src": add_out,
                "dst": scale_out,
                "elements": rows * cols,
                "scale": 0.5,
                "bias": -1.0,
            },
            {
                "op": "RMS_NORM_F32",
                "src": scale_out,
                "dst": norm_out,
                "rows": rows,
                "cols": cols,
                "eps": 1.0e-6,
            },
            {
                "op": "SILU_F32",
                "src": norm_out,
                "dst": silu_out,
                "elements": rows * cols,
            },
            {
                "op": "MUL_F32",
                "src0": silu_out,
                "src1": row_bias,
                "dst": mul_out,
                "rows": rows,
                "cols": cols,
                "src1_broadcast": True,
            },
        ],
        outputs=[mul_out],
    )

    assert payload == b""
    assert_counters(response["result"], {
        "ps_ops": 5,
        "pl_ops": 0,
        "bytes_read": (8 + 4 + 8 + 8 + 8 + 8 + 4) * 4,
        "bytes_written": 5 * 8 * 4,
    })

    add = []
    for col in range(cols):
        for row in range(rows):
            add.append(values[col * rows + row] + bias[row])
    scaled = [(value * 0.5) - 1.0 for value in add]
    normed = []
    for col in range(cols):
        chunk = scaled[col * rows : (col + 1) * rows]
        scale = 1.0 / ((sum(value * value for value in chunk) / rows + 1.0e-6) ** 0.5)
        normed.extend(value * scale for value in chunk)
    silu = [value / (1.0 + math.exp(-value)) for value in normed]
    expected = []
    for col in range(cols):
        for row in range(rows):
            expected.append(silu[col * rows + row] * bias[row])

    _, payload = client.call("DOWNLOAD_TENSOR", handle=mul_out, size=nbytes)
    assert struct.unpack("<8f", payload) == pytest.approx(expected)


def test_run_graph_swiglu_f32_can_alias_gate(client):
    values = [0.5, -1.0, 2.0, -4.0, 1.5, -2.0, 3.0, -6.0]
    up = [1.0, 0.5, -0.25, 2.0, 1.5, -0.5, 0.25, -2.0]
    nbytes = len(values) * 4

    response, _ = client.call("ALLOC_TENSOR", nbytes=2 * nbytes, dtype="F32")
    arena = response["result"]["tensor"]["handle"]
    client.call("UPLOAD_TENSOR", payload=struct.pack("<8f", *values), handle=arena)
    client.call(
        "UPLOAD_TENSOR",
        payload=struct.pack("<8f", *up),
        handle=arena,
        offset=nbytes,
    )

    response, payload = client.call(
        "RUN_GRAPH",
        graph_version=1,
        ops=[
            {
                "op": "SWIGLU_F32",
                "src0": arena,
                "src1": arena,
                "dst": arena,
                "elements": len(values),
                "src0_offset": 0,
                "src1_offset": nbytes,
                "dst_offset": 0,
            },
        ],
        outputs=[arena],
    )

    assert payload == b""
    assert_counters(response["result"], {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": 2 * nbytes,
        "bytes_written": nbytes,
    })

    _, payload = client.call("DOWNLOAD_TENSOR", handle=arena, size=nbytes)
    expected = [
        (value / (1.0 + math.exp(-value))) * up_value
        for value, up_value in zip(values, up)
    ]
    assert struct.unpack("<8f", payload) == pytest.approx(expected)


def test_run_graph_profile_emits_per_op_json(monkeypatch, capsys):
    allocator = fake_allocator(total_bytes=4096, slab_bytes=4096)
    src = allocator.allocate(16, usage="profile-src").handle
    dst = allocator.allocate(16, usage="profile-dst").handle
    allocator.write(src, 0, b"0123456789abcdef")
    monkeypatch.setenv("PYNQ_PROFILE", "1")

    result = run_graph(
        allocator,
        {
            "graph_version": 1,
            "ops": [
                {
                    "op": "COPY",
                    "name": "copy-profile",
                    "src": src,
                    "dst": dst,
                    "nbytes": 16,
                },
            ],
            "outputs": [dst],
        },
    )

    assert result["counters"]["bytes_read"] == 16
    captured = capsys.readouterr()
    lines = [
        line.removeprefix("pynq profile: ")
        for line in captured.err.splitlines()
        if line.startswith("pynq profile: ")
    ]
    assert len(lines) == 1

    profile = json.loads(lines[0])
    assert profile["graph_us"] >= 0
    assert profile["counters"]["bytes_read"] == 16
    assert profile["counters"]["bytes_written"] == 16
    assert profile["ops"] == [
        {
            "index": 0,
            "op": "COPY",
            "read_us": profile["ops"][0]["read_us"],
            "compute_us": 0,
            "write_us": profile["ops"][0]["write_us"],
            "total_us": profile["ops"][0]["total_us"],
            "bytes_read": 16,
            "bytes_written": 16,
            "name": "copy-profile",
            "nbytes": 16,
        },
    ]


def test_unsupported_graph_op_is_reported(client):
    with pytest.raises(RpcRemoteError) as exc:
        client.call("RUN_GRAPH", graph_version=1, ops=[{"op": "MATMUL"}])

    assert exc.value.code == "unsupported_op"
