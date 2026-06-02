"""Per-graph span recorder and event emitter.

A ``Timer`` is constructed for one ``RUN_GRAPH`` call. Code records spans
with ``timer.section(name)`` and ``timer.op(name)``; the timer accumulates
totals AND emits structured events via ``board.profiling.events`` so an
external analyzer can reconstruct timing without inspecting Python state.

Kernels stay agnostic of profiling — they just call ``timer.section(...)``
and ``timer.add(...)``. The emission machinery lives here.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from board.profiling import events

# Reused no-op context for section()/op() on the disabled path — avoids
# building a fresh generator-based context manager per (sub-)op.
_NULL_CTX = nullcontext()


@dataclass
class OpSpan:
    op: str
    index: int
    fields: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, int] = field(default_factory=dict)
    total_us: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "index": self.index,
            "total_us": self.total_us,
            **self.sections,
            **self.fields,
        }


class Timer:
    """Per-graph span recorder.

    When ``profile`` is off (the default in production, gated on
    ``PYNQ_PROFILE``), op()/section() are near-no-ops: no per-op span object,
    no event emission, no per-section ``perf_counter`` — only the two byte
    counters the RUN_GRAPH response needs are accumulated. On the A9 the span +
    event + section machinery cost ~0.2-0.3ms per op × ~40k ops/token; skipping
    it when nobody reads the NDJSON is the win. With ``profile`` on, the full
    per-op spans + ``op_begin``/``op_end`` events are recorded as before.
    """

    def __init__(self, req_id: int | None = None, *, profile: bool | None = None) -> None:
        self._ops: list[OpSpan] = []
        self._current: OpSpan | None = None
        self.graph_us: int = 0
        self.req_id = req_id
        self._profile = events.enabled() if profile is None else profile
        # Always tracked (cheap) so the response counters don't need spans.
        self.total_bytes_read: int = 0
        self.total_bytes_written: int = 0

    @property
    def profile(self) -> bool:
        return self._profile

    @contextmanager
    def _op_profiled(self, name: str, **fields: Any) -> Iterator[OpSpan]:
        fields.pop("index", None)  # span owns its own index; ignore any duplicate
        span = OpSpan(op=name, index=len(self._ops), fields=dict(fields))
        previous = self._current
        self._current = span
        events.emit("op_begin", req_id=self.req_id, op=name, index=span.index, **fields)
        start_ns = time.perf_counter_ns()
        try:
            yield span
        finally:
            span.total_us = _us(start_ns)
            self._ops.append(span)
            self._current = previous
            events.emit(
                "op_end",
                req_id=self.req_id,
                op=span.op,
                index=span.index,
                total_us=span.total_us,
                **span.sections,
                **span.fields,
            )

    def op(self, name: str, **fields: Any):
        """Record one graph op span (profiled), or a no-op context (disabled)."""
        if self._profile:
            return self._op_profiled(name, **fields)
        return _NULL_CTX

    def section(self, name: str):
        """Record a named sub-span inside the current op (profiled only)."""
        if not self._profile:
            return _NULL_CTX
        return self._section_profiled(name)

    @contextmanager
    def _section_profiled(self, name: str) -> Iterator[None]:
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = _us(start_ns)
            if self._current is None:
                self.graph_us += elapsed
            else:
                key = f"{name}_us"
                self._current.sections[key] = self._current.sections.get(key, 0) + elapsed

    def add(self, key: str, value: Any) -> None:
        """Accumulate a field. The two byte counters always feed the graph-level
        totals (used by the response); the rest land on the current span when
        profiling."""
        if key == "bytes_read":
            self.total_bytes_read += value
        elif key == "bytes_written":
            self.total_bytes_written += value
        if self._current is not None:
            existing = self._current.fields.get(key, 0)
            self._current.fields[key] = existing + value

    def summary(self, counters: dict[str, int] | None = None) -> dict[str, Any]:
        """Aggregate payload — matches the legacy ``pynq profile:`` JSON shape."""
        return {
            "graph_us": self.graph_us,
            "counters": counters or {},
            "ops": [span.to_dict() for span in self._ops],
        }

    @property
    def ops(self) -> list[OpSpan]:
        return list(self._ops)


def _us(start_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - start_ns) // 1000)
