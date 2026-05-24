"""End-to-end tests for things that *require* the real wire.

Component-level allocator/kernel/graph coverage lives in ``board/tests/``;
this file only asserts behavior that depends on framing, threading, or
the JSON envelope round-tripping through a real TCP connection.
"""

from __future__ import annotations

import socket

import pytest

from host.transport.client import RpcClient, RpcRemoteError
from proto.ops import (
    GOP_COPY,
    GRAPH_VERSION,
    OP_ALLOC_TENSOR,
    OP_DOWNLOAD_TENSOR,
    OP_FREE_TENSOR,
    OP_HELLO,
    OP_MEMORY,
    OP_RUN_GRAPH,
    OP_UPLOAD_TENSOR,
)
from tests.golden.vectors import deterministic_bytes


def test_hello_reports_capabilities(rpc_client):
    response, _ = rpc_client.call(OP_HELLO)
    result = response["result"]
    assert result["abi_version"] == 1
    assert result["server"] == "bonsaid"
    assert result["overlay_id"] == "test-overlay"
    assert OP_ALLOC_TENSOR in result["capabilities"]
    assert OP_RUN_GRAPH in result["capabilities"]
    assert GOP_COPY in result["graph_ops"]


def test_memory_reports_zero_used_at_startup(rpc_client):
    response, _ = rpc_client.call(OP_MEMORY)
    memory = response["result"]["memory"]
    assert memory["used_bytes"] == 0
    assert memory["tensor_count"] == 0


def test_upload_download_round_trip_over_wire(rpc_client):
    nbytes = 64 * 1024
    payload = deterministic_bytes(nbytes)

    alloc, _ = rpc_client.call(OP_ALLOC_TENSOR, nbytes=nbytes)
    handle = alloc["result"]["tensor"]["handle"]

    rpc_client.call(OP_UPLOAD_TENSOR, payload=payload, handle=handle, offset=0)
    _, downloaded = rpc_client.call(
        OP_DOWNLOAD_TENSOR, handle=handle, offset=0, size=nbytes
    )
    assert downloaded == payload

    rpc_client.call(OP_FREE_TENSOR, handle=handle)


def test_run_graph_copy_over_wire(rpc_client):
    nbytes = 4 * 1024
    src_alloc, _ = rpc_client.call(OP_ALLOC_TENSOR, nbytes=nbytes)
    dst_alloc, _ = rpc_client.call(OP_ALLOC_TENSOR, nbytes=nbytes)
    src = src_alloc["result"]["tensor"]["handle"]
    dst = dst_alloc["result"]["tensor"]["handle"]

    rpc_client.call(OP_UPLOAD_TENSOR, payload=deterministic_bytes(nbytes), handle=src)
    rpc_client.call(
        OP_RUN_GRAPH,
        graph_version=GRAPH_VERSION,
        ops=[{"op": GOP_COPY, "src": src, "dst": dst, "nbytes": nbytes}],
        outputs=[dst],
    )

    _, payload = rpc_client.call(OP_DOWNLOAD_TENSOR, handle=dst, offset=0, size=nbytes)
    assert payload == deterministic_bytes(nbytes)


def test_remote_error_propagates_typed(rpc_client):
    with pytest.raises(RpcRemoteError) as exc:
        rpc_client.call(OP_DOWNLOAD_TENSOR, handle=999, offset=0, size=1)
    assert exc.value.code == "unknown_tensor"


def test_unsupported_op_returns_typed_error(rpc_client):
    with pytest.raises(RpcRemoteError) as exc:
        rpc_client.call("NOPE")
    assert exc.value.code == "unsupported_op"


def test_multiple_requests_share_one_connection(rpc_client):
    """Verify the server handles a sequence of requests on one socket."""
    for _ in range(5):
        response, _ = rpc_client.call(OP_HELLO)
        assert response["ok"]


def test_concurrent_connections(rpc_server):
    """Two clients can drive the same daemon at once."""
    def hammer():
        with socket.create_connection(rpc_server.server_address, timeout=2) as sock:
            client = RpcClient(sock)
            for _ in range(3):
                client.call(OP_HELLO)

    import threading
    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()
