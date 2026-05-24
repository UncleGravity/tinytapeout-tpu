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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from board.profiling import events


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
    def __init__(self, req_id: int | None = None) -> None:
        self._ops: list[OpSpan] = []
        self._current: OpSpan | None = None
        self.graph_us: int = 0
        self.req_id = req_id

    @contextmanager
    def op(self, name: str, **fields: Any) -> Iterator[OpSpan]:
        """Record one graph op span. Emits ``op_begin`` / ``op_end`` events."""
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

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Record a named sub-span inside the current op."""
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
        """Accumulate a free-form field on the current op span."""
        if self._current is None:
            return
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
