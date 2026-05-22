from __future__ import annotations

import socket
import struct
import threading

import pytest

from runtime.allocator import fake_allocator
from runtime.bonsai_rpc import RpcClient, RpcRemoteError
from runtime.bonsaid import BonsaiRpcServer, BonsaiRuntime


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
    assert result["graph_ops"] == ["COPY", "MATMUL_Q1A8"]


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
    assert result["counters"] == {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": nbytes,
        "bytes_written": nbytes,
    }

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
    assert response["result"]["counters"] == {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": len(weights) + len(acts),
        "bytes_written": rows * cols * 4,
    }

    _, payload = client.call("DOWNLOAD_TENSOR", handle=dst_handle, size=16)
    got = struct.unpack("<4f", payload)
    q8_scale = struct.unpack("<e", struct.pack("<e", 1.0 / 127.0))[0]
    assert got == pytest.approx((0.0, 0.0, 128.0 * 127.0 * q8_scale, 0.0))


def test_unsupported_graph_op_is_reported(client):
    with pytest.raises(RpcRemoteError) as exc:
        client.call("RUN_GRAPH", graph_version=1, ops=[{"op": "MATMUL"}])

    assert exc.value.code == "unsupported_op"
