from __future__ import annotations

import socket
import threading

import pytest

from host.transport.client import RpcClient
from runtime.allocator import fake_allocator
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
def rpc_client(rpc_server):
    with socket.create_connection(rpc_server.server_address, timeout=2) as sock:
        yield RpcClient(sock)
