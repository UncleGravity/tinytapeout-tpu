"""RpcClient correlation + error unwrap, using a socketpair as the wire."""

from __future__ import annotations

import socket
import threading

import pytest

from host.transport.client import RpcClient, RpcRemoteError
from proto.framing import ProtocolError, recv_message, send_message


def _start_responder(sock: socket.socket, responder):
    def loop():
        try:
            while True:
                metadata, payload = recv_message(sock)
                response, response_payload = responder(metadata, payload)
                if response is None:
                    return
                send_message(sock, response, response_payload)
        except (EOFError, OSError):
            return

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def test_call_round_trip():
    a, b = socket.socketpair()
    try:
        def responder(metadata, payload):
            return {"id": metadata["id"], "ok": True, "result": {"echo": metadata["op"]}}, b""

        _start_responder(b, responder)
        client = RpcClient(a)
        response, payload = client.call("HELLO")
        assert response["result"] == {"echo": "HELLO"}
        assert payload == b""
    finally:
        a.close()
        b.close()


def test_remote_error_raises_typed_exception():
    a, b = socket.socketpair()
    try:
        def responder(metadata, payload):
            return ({
                "id": metadata["id"],
                "ok": False,
                "error": {"code": "out_of_memory", "message": "nope"},
            }, b"")

        _start_responder(b, responder)
        client = RpcClient(a)
        with pytest.raises(RpcRemoteError) as exc:
            client.call("ALLOC_TENSOR", nbytes=1)
        assert exc.value.code == "out_of_memory"
        assert "nope" in exc.value.message
    finally:
        a.close()
        b.close()


def test_id_mismatch_raises_protocol_error():
    a, b = socket.socketpair()
    try:
        def responder(metadata, _payload):
            # Reply with a different id on purpose.
            return {"id": metadata["id"] + 99, "ok": True, "result": {}}, b""

        _start_responder(b, responder)
        client = RpcClient(a)
        with pytest.raises(ProtocolError):
            client.call("HELLO")
    finally:
        a.close()
        b.close()


def test_sequential_calls_use_monotonic_ids():
    a, b = socket.socketpair()
    seen = []
    try:
        def responder(metadata, _payload):
            seen.append(metadata["id"])
            return {"id": metadata["id"], "ok": True, "result": {}}, b""

        _start_responder(b, responder)
        client = RpcClient(a)
        client.call("HELLO")
        client.call("HELLO")
        client.call("HELLO")
        assert seen == [1, 2, 3]
    finally:
        a.close()
        b.close()


def test_payload_round_trips():
    a, b = socket.socketpair()
    try:
        def responder(metadata, payload):
            return ({"id": metadata["id"], "ok": True, "result": {}}, payload)

        _start_responder(b, responder)
        client = RpcClient(a)
        _, payload = client.call("UPLOAD", payload=b"\x01\x02\x03" * 1000)
        assert payload == b"\x01\x02\x03" * 1000
    finally:
        a.close()
        b.close()
