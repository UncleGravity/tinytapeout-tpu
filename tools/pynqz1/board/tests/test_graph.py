"""``run_graph`` dispatch with an in-process registry — no sockets."""

from __future__ import annotations

import pytest

from board.daemon.graph import run_graph
from board.memory.allocator import AllocatorError
from proto.ops import GOP_COPY, GRAPH_VERSION
from tests.golden.vectors import deterministic_bytes


def allocate_pair(allocator, nbytes):
    src = allocator.allocate(nbytes, shape=[nbytes], dtype="u8")
    dst = allocator.allocate(nbytes, shape=[nbytes], dtype="u8")
    allocator.write(src.handle, 0, deterministic_bytes(nbytes))
    return src.handle, dst.handle


def test_run_graph_copy_round_trip(allocator, registry):
    src, dst = allocate_pair(allocator, 4096)
    result = run_graph(allocator, registry, {
        "graph_version": GRAPH_VERSION,
        "ops": [{"op": GOP_COPY, "src": src, "dst": dst, "nbytes": 4096}],
        "outputs": [dst],
    })

    assert result["op_count"] == 1
    assert result["outputs"] == [dst]
    counters = result["counters"]
    assert counters["ps_ops"] == 1
    assert counters["bytes_read"] == 4096
    assert counters["bytes_written"] == 4096
    assert counters["elapsed_us"] >= 0

    assert allocator.read(dst, 0, 4096) == deterministic_bytes(4096)


def test_run_graph_rejects_wrong_version(allocator, registry):
    with pytest.raises(AllocatorError) as exc:
        run_graph(allocator, registry, {
            "graph_version": GRAPH_VERSION + 1,
            "ops": [],
            "outputs": [],
        })
    assert exc.value.code == "unsupported_graph_version"


def test_run_graph_rejects_unknown_op(allocator, registry):
    with pytest.raises(AllocatorError) as exc:
        run_graph(allocator, registry, {
            "graph_version": GRAPH_VERSION,
            "ops": [{"op": "NOPE"}],
            "outputs": [],
        })
    assert exc.value.code == "unsupported_op"


def test_run_graph_validates_outputs_exist(allocator, registry):
    src, dst = allocate_pair(allocator, 16)
    with pytest.raises(AllocatorError) as exc:
        run_graph(allocator, registry, {
            "graph_version": GRAPH_VERSION,
            "ops": [{"op": GOP_COPY, "src": src, "dst": dst, "nbytes": 16}],
            "outputs": [9999],
        })
    assert exc.value.code == "unknown_tensor"
