"""Per-graph timing without leaking into kernel signatures.

A ``Timer`` is constructed for the duration of one ``RUN_GRAPH`` request.
Kernels and dispatch code record spans via ``timer.section(name)``; the
recorded spans are emitted by the caller after the graph completes.

Example
-------

    timer = Timer()
    for op in ops:
        with timer.op(op["op"]):
            kernel.run(allocator, op, timer)
    timer.emit_if_enabled()

Inside a kernel:

    def run(self, allocator, op, timer):
        with timer.section("read"):
            data = allocator.read(...)
        with timer.section("compute"):
            self._fn(...)
        with timer.section("write"):
            allocator.write(...)
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


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
    def __init__(self) -> None:
        self._ops: list[OpSpan] = []
        self._current: OpSpan | None = None
        self.graph_us: int = 0

    @contextmanager
    def op(self, name: str, **fields: Any) -> Iterator[OpSpan]:
        """Record one graph op span. Pushes a new ``OpSpan`` as current."""
        span = OpSpan(op=name, index=len(self._ops), fields=dict(fields))
        previous = self._current
        self._current = span
        start_ns = time.perf_counter_ns()
        try:
            yield span
        finally:
            span.total_us = _us(start_ns)
            self._ops.append(span)
            self._current = previous

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Record a named sub-span inside the current op."""
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed = _us(start_ns)
            if self._current is None:
                # Stray section outside an op — accumulate at the top level.
                self.graph_us += elapsed
            else:
                self._current.sections[f"{name}_us"] = (
                    self._current.sections.get(f"{name}_us", 0) + elapsed
                )

    def add(self, key: str, value: Any) -> None:
        """Accumulate a free-form field on the current op span."""
        if self._current is None:
            return
        existing = self._current.fields.get(key, 0)
        self._current.fields[key] = existing + value

    def emit_if_enabled(self, *, counters: dict[str, int] | None = None) -> None:
        if not _profile_enabled():
            return
        payload = {
            "graph_us": self.graph_us,
            "counters": counters or {},
            "ops": [span.to_dict() for span in self._ops],
        }
        print(
            "pynq profile: " + json.dumps(payload, separators=(",", ":")),
            file=sys.stderr,
            flush=True,
        )

    @property
    def ops(self) -> list[OpSpan]:
        return list(self._ops)


def _us(start_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - start_ns) // 1000)


def _profile_enabled() -> bool:
    value = os.environ.get("PYNQ_PROFILE")
    return value is not None and value.lower() not in ("", "0", "false", "no", "off")
