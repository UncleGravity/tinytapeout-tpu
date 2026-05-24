"""Kernel contract used by the daemon to execute graph ops.

A ``Kernel`` is the single swap-point between the daemon and a compute
implementation. The same interface backs the PS kernels today and will
back the PL kernel once the overlay is in place — adding a new backend
is one ``registry.register(...)`` call.

The kernel sees the allocator and the op dict, plus a ``Timer`` it uses
to record ``read`` / ``compute`` / ``write`` spans. Kernels are free to
pull bytes through the allocator (PS path) or to look up extent physical
addresses for DMA (future PL path) — the daemon imposes neither.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from board.memory.allocator import TensorAllocator
from board.profiling.timer import Timer


@runtime_checkable
class Kernel(Protocol):
    name: str

    def run(self, allocator: TensorAllocator, op: dict[str, Any], timer: Timer) -> None: ...
