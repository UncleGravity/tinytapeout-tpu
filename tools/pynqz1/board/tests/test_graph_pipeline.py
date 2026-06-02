"""Pipelining scheduler in run_graph: overlap ordering + dependency safety.

Uses fake kernels (no native lib, no board) to assert the single-in-flight
scheduler issues a matmul, runs independent PS ops during its DMA, and
completes it exactly before an op that consumes its result or before another
matmul needs the kernel.
"""

from __future__ import annotations

from board.daemon.graph import _op_touches, run_graph
from board.kernels.registry import KernelRegistry
from proto.ops import (
    F_COLS,
    F_DST,
    F_DST_OFFSET,
    F_ROWS,
    F_SRC,
    F_SRC_OFFSET,
    GOP_MATMUL_Q1A8,
    GRAPH_VERSION,
)


class _Pending:
    def __init__(self, dst_handle: int, dst_lo: int, dst_hi: int):
        self.dst_handle = dst_handle
        self.dst_lo = dst_lo
        self.dst_hi = dst_hi
        self.op_name = None
        self.kernel = None


class FakeMatmul:
    name = GOP_MATMUL_Q1A8
    backend = "pl"

    def __init__(self, log: list, pipelinable: bool = True):
        self.log = log
        self.pipelinable = pipelinable

    def run_async(self, allocator, op, timer):
        if not self.pipelinable:
            self.log.append(("run", op[F_DST]))
            return None
        self.log.append(("issue", op[F_DST]))
        lo = int(op.get(F_DST_OFFSET, 0))
        hi = lo + int(op[F_ROWS]) * int(op[F_COLS]) * 4
        return _Pending(int(op[F_DST]), lo, hi)

    def complete(self, pending, timer):
        self.log.append(("complete", pending.dst_handle))

    def run(self, allocator, op, timer):  # sync fallback path
        self.log.append(("run", op[F_DST]))


class FakePS:
    backend = "ps"

    def __init__(self, name: str, log: list):
        self.name = name
        self.log = log

    def run(self, allocator, op, timer):
        self.log.append(("run", self.name))


def _registry(log, pipelinable=True):
    reg = KernelRegistry()
    reg.register(FakeMatmul(log, pipelinable=pipelinable))
    reg.register(FakePS("PS_A", log))
    reg.register(FakePS("PS_B", log))
    return reg


def _run(reg, ops):
    return run_graph(None, reg, {
        "graph_version": GRAPH_VERSION,
        "ops": ops,
        "outputs": [],
    })


def _mm(dst, rows=1, cols=1, dst_offset=0):
    return {"op": GOP_MATMUL_Q1A8, F_DST: dst, F_DST_OFFSET: dst_offset,
            F_ROWS: rows, F_COLS: cols}


def _ps(name, src, dst):
    return {"op": name, F_SRC: src, F_SRC_OFFSET: 0, F_DST: dst, F_DST_OFFSET: 0}


# -- _op_touches unit ------------------------------------------------------


def test_op_touches_offset_interval():
    # matmul dst = handle 7, bytes [0, 64)
    assert _op_touches(_ps("PS_A", src=7, dst=9), 7, 0, 64)          # src reads it
    assert _op_touches({"op": "X", F_DST: 7, F_DST_OFFSET: 32}, 7, 0, 64)  # dst inside
    assert not _op_touches(_ps("PS_A", src=8, dst=9), 7, 0, 64)      # other handle
    # offset at/after the end of the interval does not touch it
    assert not _op_touches({"op": "X", F_SRC: 7, F_SRC_OFFSET: 64}, 7, 0, 64)
    assert _op_touches({"op": "X", F_SRC: 7, F_SRC_OFFSET: 63}, 7, 0, 64)


# -- scheduler ordering ----------------------------------------------------


def test_independent_op_overlaps_then_dependent_forces_complete():
    log: list = []
    reg = _registry(log)
    # matmul → H1 ; PS_A reads H2 (independent) ; PS_B reads H1 (consumer)
    _run(reg, [_mm(dst=1, rows=4, cols=4),
               _ps("PS_A", src=2, dst=3),
               _ps("PS_B", src=1, dst=4)])
    assert log == [
        ("issue", 1),       # matmul DMA started
        ("run", "PS_A"),    # independent PS runs during the stream
        ("complete", 1),    # consumer forces the wait
        ("run", "PS_B"),
    ]


def test_second_matmul_forces_complete():
    log: list = []
    reg = _registry(log)
    _run(reg, [_mm(dst=1), _mm(dst=2)])
    # one kernel → first must finish before the second issues
    assert log == [("issue", 1), ("complete", 1), ("issue", 2), ("complete", 2)]


def test_lone_matmul_drains_at_end():
    log: list = []
    reg = _registry(log)
    _run(reg, [_mm(dst=1)])
    assert log == [("issue", 1), ("complete", 1)]


def test_dependency_uses_offset_not_just_handle():
    log: list = []
    reg = _registry(log)
    # matmul writes H1 bytes [0,16) (rows*cols*4 = 1*4*4=16). A PS op reading
    # H1 at offset 64 is a *different* tensor in the same arena buffer → may
    # overlap the stream; a PS op reading H1 at offset 0 must wait.
    _run(reg, [_mm(dst=1, rows=1, cols=4, dst_offset=0),
               _ps("PS_A", src=1, dst=2) | {F_SRC_OFFSET: 64},
               _ps("PS_B", src=1, dst=3) | {F_SRC_OFFSET: 0}])
    assert log == [
        ("issue", 1),
        ("run", "PS_A"),    # offset 64 is outside [0,16) → overlaps
        ("complete", 1),    # offset 0 consumes the result → wait
        ("run", "PS_B"),
    ]


def test_sync_fallback_when_not_pipelinable():
    log: list = []
    reg = _registry(log, pipelinable=False)
    _run(reg, [_mm(dst=1), _ps("PS_A", src=1, dst=2)])
    # run_async returned None → matmul ran synchronously, no overlap
    assert log == [("run", 1), ("run", "PS_A")]


def test_counters_count_pl_and_ps():
    log: list = []
    reg = _registry(log)
    result = _run(reg, [_mm(dst=1, rows=4, cols=4),
                        _ps("PS_A", src=2, dst=3),
                        _ps("PS_B", src=1, dst=4)])
    assert result["counters"]["pl_ops"] == 1
    assert result["counters"]["ps_ops"] == 2
    assert result["op_count"] == 3
