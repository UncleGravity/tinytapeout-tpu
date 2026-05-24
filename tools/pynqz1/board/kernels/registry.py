"""Op-name → kernel lookup. Populated at daemon startup."""

from __future__ import annotations

from board.kernels.interface import Kernel
from board.memory.allocator import AllocatorError


class KernelRegistry:
    def __init__(self) -> None:
        self._kernels: dict[str, Kernel] = {}

    def register(self, kernel: Kernel) -> None:
        if not isinstance(kernel, Kernel):
            raise TypeError(f"{kernel!r} does not implement the Kernel protocol")
        self._kernels[kernel.name] = kernel

    def get(self, op_name: str) -> Kernel:
        try:
            return self._kernels[op_name]
        except KeyError:
            raise AllocatorError(
                "unsupported_op", f"unsupported graph op {op_name}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._kernels)

    def __contains__(self, op_name: str) -> bool:
        return op_name in self._kernels
