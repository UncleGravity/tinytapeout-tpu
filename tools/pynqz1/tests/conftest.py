"""Fixtures for end-to-end tests that spin up a real daemon on localhost."""

from __future__ import annotations

import socket
import subprocess
import threading
from pathlib import Path

import pytest

from board.daemon.runtime import Runtime
from board.daemon.server import BonsaiRpcServer
from board.kernels.ps import native as ps_native
from board.kernels.registry import KernelRegistry
from board.memory.allocator import TensorAllocator
from board.memory.slabs import fake_slabs
from host.transport.client import RpcClient


PS_DIR = Path(__file__).resolve().parents[1] / "board" / "kernels" / "ps"


@pytest.fixture(scope="session")
def native_lib_path(tmp_path_factory) -> Path:
    """Build libbonsai_ps.so into a session tmpdir, never into the source tree."""
    out_dir = tmp_path_factory.mktemp("ps_native")
    subprocess.run(
        ["make", "-C", str(PS_DIR), f"OUT_DIR={out_dir}"],
        check=True,
    )
    return out_dir / "libbonsai_ps.so"


@pytest.fixture
def rpc_server(native_lib_path):
    allocator = TensorAllocator(fake_slabs(total_bytes=2 << 20, slab_bytes=512 * 1024))
    registry = KernelRegistry()
    ps_native.register_all(registry, lib_path=native_lib_path)
    runtime = Runtime(allocator, registry, overlay_id="test-overlay")

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
