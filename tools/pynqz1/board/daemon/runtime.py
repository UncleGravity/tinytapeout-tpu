"""``Runtime`` holds the daemon's shared mutable state.

Owned: allocator + kernel registry + overlay handle + a single lock that
serializes every request (the kernel layer is not yet thread-safe). The
server module dispatches each incoming frame through ``Runtime.dispatch``.
"""

from __future__ import annotations

import threading
from typing import Any

from board.daemon.graph import run_graph
from board.kernels.registry import KernelRegistry
from board.memory.allocator import AllocatorError, TensorAllocator
from proto.ops import (
    ABI_VERSION,
    F_ABI_VERSION,
    F_ALIGNMENT,
    F_CAPABILITIES,
    F_DTYPE,
    F_GRAPH_OPS,
    F_HANDLE,
    F_LAYOUT,
    F_MEMORY,
    F_NBYTES,
    F_OFFSET,
    F_OP,
    F_OVERLAY_ID,
    F_SERVER,
    F_SHAPE,
    F_SIZE,
    F_TENSOR,
    F_USAGE,
    OP_ALLOC_TENSOR,
    OP_DOWNLOAD_TENSOR,
    OP_FREE_TENSOR,
    OP_HELLO,
    OP_MEMORY,
    OP_RUN_GRAPH,
    OP_UPLOAD_TENSOR,
    RPC_OPS,
    SERVER_NAME,
)


class Runtime:
    def __init__(
        self,
        allocator: TensorAllocator,
        registry: KernelRegistry,
        overlay_id: str,
        overlay: object | None = None,
    ):
        self.allocator = allocator
        self.registry = registry
        self.overlay_id = overlay_id
        self.overlay = overlay
        self._lock = threading.Lock()

    def dispatch(
        self, metadata: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        op = str(metadata.get(F_OP, ""))
        if not op:
            raise AllocatorError("invalid_request", "missing op")

        with self._lock:
            return self._handle(op, metadata, payload)

    def close(self) -> None:
        self.allocator.close()

    # -- handlers ---------------------------------------------------------

    def _handle(
        self, op: str, metadata: dict[str, Any], payload: bytes
    ) -> tuple[dict[str, Any], bytes]:
        if op == OP_HELLO:
            return self._hello(), b""

        if op == OP_MEMORY:
            return {F_MEMORY: self.allocator.memory_info()}, b""

        if op == OP_ALLOC_TENSOR:
            record = self.allocator.allocate(
                _required_int(metadata, F_NBYTES),
                shape=_optional_list(metadata, F_SHAPE),
                dtype=str(metadata.get(F_DTYPE, "u8")),
                usage=str(metadata.get(F_USAGE, "tensor")),
                layout=str(metadata.get(F_LAYOUT, "raw")),
                alignment=int(metadata.get(F_ALIGNMENT, 64)),
            )
            return {F_TENSOR: self.allocator.describe(record.handle)}, b""

        if op == OP_UPLOAD_TENSOR:
            handle = _required_int(metadata, F_HANDLE)
            offset = int(metadata.get(F_OFFSET, 0))
            self.allocator.write(handle, offset, payload)
            return {
                F_TENSOR: self.allocator.describe(handle),
                "written": len(payload),
            }, b""

        if op == OP_DOWNLOAD_TENSOR:
            handle = _required_int(metadata, F_HANDLE)
            offset = int(metadata.get(F_OFFSET, 0))
            size = _required_int(metadata, F_SIZE)
            data = self.allocator.read(handle, offset, size)
            return {
                F_TENSOR: self.allocator.describe(handle),
                "read": len(data),
            }, data

        if op == OP_FREE_TENSOR:
            handle = _required_int(metadata, F_HANDLE)
            record = self.allocator.free(handle)
            return {
                "freed": {"handle": record.handle, F_NBYTES: record.nbytes},
                F_MEMORY: self.allocator.memory_info(),
            }, b""

        if op == OP_RUN_GRAPH:
            return run_graph(self.allocator, self.registry, metadata), b""

        raise AllocatorError("unsupported_op", f"unsupported op {op}")

    def _hello(self) -> dict[str, Any]:
        return {
            F_ABI_VERSION: ABI_VERSION,
            F_SERVER: SERVER_NAME,
            F_OVERLAY_ID: self.overlay_id,
            F_MEMORY: self.allocator.memory_info(),
            F_CAPABILITIES: list(RPC_OPS),
            F_GRAPH_OPS: self.registry.names(),
        }


# -- helpers ---------------------------------------------------------------


def _required_int(metadata: dict[str, Any], key: str) -> int:
    if key not in metadata:
        raise AllocatorError("invalid_request", f"missing {key}")
    value = int(metadata[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


def _optional_list(metadata: dict[str, Any], key: str) -> list[int]:
    value = metadata.get(key, [])
    if not isinstance(value, list):
        raise AllocatorError("invalid_request", f"{key} must be a list")
    return [int(item) for item in value]
