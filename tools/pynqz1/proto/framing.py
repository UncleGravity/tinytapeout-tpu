"""Wire framing for the PYNQ-Z1 runtime protocol.

Frame layout (matches host C++ in ``host/transport/client.cpp``):

    u8  magic[4]    "BPNQ"
    u16 version     big endian, currently 1
    u16 flags       big endian, 0
    u32 json_len    big endian
    u32 payload_len big endian
    u8  json[json_len]    UTF-8 JSON object
    u8  payload[payload_len]

Only the framing primitives live here. Client-side request/response
correlation belongs to ``host/transport/client.py``; server-side handler
dispatch belongs to the board daemon.
"""

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
    """Raised when a frame is malformed or violates the wire contract."""


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
    metadata_bytes = json.dumps(
        metadata, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
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
