from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAGIC = b"BPNQ"
VERSION = 1
HEADER = struct.Struct("!4sHHII")
MAX_JSON_BYTES = 1 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class RpcRemoteError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def read_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < nbytes:
        chunk = sock.recv(nbytes - len(chunks))
        if not chunk:
            if not chunks:
                raise EOFError
            raise ProtocolError("unexpected EOF while reading frame")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(
    sock: socket.socket,
    metadata: dict[str, Any],
    payload: bytes | bytearray | memoryview = b"",
) -> None:
    metadata_bytes = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    payload_view = memoryview(payload)
    header = HEADER.pack(MAGIC, VERSION, 0, len(metadata_bytes), len(payload_view))
    sock.sendall(header)
    sock.sendall(metadata_bytes)
    if payload_view:
        sock.sendall(payload_view)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    magic, version, _flags, json_len, payload_len = HEADER.unpack(
        read_exact(sock, HEADER.size)
    )
    if magic != MAGIC:
        raise ProtocolError("bad frame magic")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if json_len > MAX_JSON_BYTES:
        raise ProtocolError(f"metadata too large: {json_len} bytes")

    metadata_raw = read_exact(sock, json_len)
    try:
        metadata = json.loads(metadata_raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid metadata JSON: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ProtocolError("metadata JSON must be an object")

    payload = read_exact(sock, payload_len) if payload_len else b""
    return metadata, payload


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

