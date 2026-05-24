"""TCP server that drives a ``Runtime``. No protocol semantics live here."""

from __future__ import annotations

import socketserver

from board.daemon.runtime import Runtime
from board.memory.allocator import AllocatorError
from proto.framing import ProtocolError, recv_message, send_message


class BonsaiRpcServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, runtime: Runtime):
        super().__init__(server_address, _Handler)
        self.runtime = runtime

    def server_close(self) -> None:
        try:
            self.runtime.close()
        finally:
            super().server_close()


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        runtime: Runtime = self.server.runtime
        while True:
            try:
                metadata, payload = recv_message(self.request)
            except EOFError:
                return
            except ProtocolError as exc:
                _send_error(self.request, None, "protocol_error", str(exc))
                return

            request_id = metadata.get("id")
            try:
                result, response_payload = runtime.dispatch(metadata, payload)
            except AllocatorError as exc:
                _send_error(self.request, request_id, exc.code, str(exc))
                continue
            except (TypeError, ValueError) as exc:
                _send_error(self.request, request_id, "invalid_request", str(exc))
                continue
            except Exception as exc:
                _send_error(self.request, request_id, "internal_error", str(exc))
                continue

            send_message(
                self.request,
                {"id": request_id, "ok": True, "result": result},
                response_payload,
            )


def _send_error(sock, request_id, code: str, message: str) -> None:
    send_message(
        sock,
        {
            "id": request_id,
            "ok": False,
            "error": {"code": code, "message": message},
        },
    )
