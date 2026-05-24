"""Host-side RPC client for the PYNQ-Z1 bonsaid runtime.

A thin wrapper around ``proto.framing`` that adds:
  - per-call monotonic request ids
  - response id correlation
  - remote error unwrapping (``ok=false`` → ``RpcRemoteError``)

Framing primitives (header layout, magic, JSON encoding) live in
``proto.framing``. This module owns nothing wire-format-specific.
"""

from __future__ import annotations

import socket
from typing import Any

from proto.framing import ProtocolError, recv_message, send_message


class RpcRemoteError(RuntimeError):
    """A non-``ok`` response from the daemon."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class RpcClient:
    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._next_id = 1

    def call(
        self,
        op: str,
        payload: bytes | bytearray | memoryview = b"",
        **fields: Any,
    ) -> tuple[dict[str, Any], bytes]:
        request_id = self._next_id
        self._next_id += 1
        metadata = {"id": request_id, "op": op, **fields}
        send_message(self._sock, metadata, payload)

        response, response_payload = recv_message(self._sock)
        if response.get("id") != request_id:
            raise ProtocolError(
                f"response id {response.get('id')} does not match request id {request_id}"
            )
        if not response.get("ok", False):
            error = response.get("error", {})
            if not isinstance(error, dict):
                raise ProtocolError("error response must contain an error object")
            raise RpcRemoteError(
                str(error.get("code", "remote_error")),
                str(error.get("message", "")),
            )
        return response, response_payload
