"""Frame round-trip + protocol-error surface."""

from __future__ import annotations

import socket
import struct
import threading

import pytest

from proto.framing import HEADER, MAGIC, ProtocolError, recv_message, send_message


def _socketpair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


def test_round_trip_metadata_only():
    a, b = _socketpair()
    try:
        send_message(a, {"op": "HELLO", "id": 7})
        metadata, payload = recv_message(b)
        assert metadata == {"op": "HELLO", "id": 7}
        assert payload == b""
    finally:
        a.close()
        b.close()


def test_round_trip_metadata_and_payload():
    a, b = _socketpair()
    try:
        send_message(a, {"op": "UPLOAD", "handle": 1}, b"abc" * 100)
        metadata, payload = recv_message(b)
        assert metadata["op"] == "UPLOAD"
        assert payload == b"abc" * 100
    finally:
        a.close()
        b.close()


def test_bad_magic_raises_protocol_error():
    a, b = _socketpair()
    try:
        bad = struct.Struct("!4sHHII").pack(b"XXXX", 1, 0, 0, 0)
        a.sendall(bad)
        with pytest.raises(ProtocolError):
            recv_message(b)
    finally:
        a.close()
        b.close()


def test_eof_before_any_bytes_raises_eof_error():
    a, b = _socketpair()
    try:
        a.close()
        with pytest.raises(EOFError):
            recv_message(b)
    finally:
        b.close()


def test_streaming_recv_completes_partial_send():
    """Mid-frame interruptions are handled by read_exact looping internally."""
    a, b = _socketpair()
    received: list = []

    def receiver():
        received.append(recv_message(b))

    t = threading.Thread(target=receiver)
    t.start()
    # Send the frame in two halves to force a second recv call on the reader.
    metadata = b'{"id":1,"op":"HELLO"}'
    header = HEADER.pack(MAGIC, 1, 0, len(metadata), 0)
    a.sendall(header)
    a.sendall(metadata)
    t.join(timeout=2)
    assert received
    assert received[0][0] == {"id": 1, "op": "HELLO"}
    a.close()
    b.close()
