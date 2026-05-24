"""Structured NDJSON event log shared by board and host.

``PYNQ_PROFILE`` controls emission:
  unset / "0" / "false"     disabled (no overhead)
  "1"                       stream events to stderr
  any other value           treated as a file path; opened append, line-buffered

Each call to ``emit`` writes one JSON object terminated by a newline. The
host side mirrors this format from ``host/backend/src/events.{h,cpp}`` so a
merged file can be ingested by ``host/cli/pynq-profile.py``.

Event schema::

    {"t_us": <wall-clock micros>, "side": "board"|"host", "kind": "<name>", ...fields}

Reserved fields:
  req_id          RPC envelope id ties host and board events together
  op              graph op name on RUN_GRAPH
  index           index of the op inside its graph
  bytes_read      bytes pulled from allocator for this op
  bytes_written   bytes written back to allocator
  us              elapsed microseconds for this event's span
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from typing import Any, TextIO

_SIDE = "board"


class _Emitter:
    def __init__(self) -> None:
        self._sink: TextIO | None = None
        self._owned = False
        self._lock = threading.Lock()
        self._configure()

    def _configure(self) -> None:
        value = os.environ.get("PYNQ_PROFILE")
        if not value or value.lower() in ("0", "false", "no", "off"):
            return
        if value == "1":
            self._sink = sys.stderr
            return
        try:
            self._sink = open(value, "a", buffering=1)
            self._owned = True
        except OSError as exc:
            print(
                f"pynq: PYNQ_PROFILE='{value}' could not be opened ({exc}); "
                f"falling back to stderr",
                file=sys.stderr,
                flush=True,
            )
            self._sink = sys.stderr

    @property
    def enabled(self) -> bool:
        return self._sink is not None

    def emit(self, kind: str, **fields: Any) -> None:
        if self._sink is None:
            return
        record = {
            "t_us": time.time_ns() // 1000,
            "side": _SIDE,
            "kind": kind,
            **fields,
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._sink.write(line + "\n")
            self._sink.flush()

    def close(self) -> None:
        if self._owned and self._sink is not None:
            try:
                self._sink.close()
            except Exception:
                pass
            self._sink = None
            self._owned = False


_emitter = _Emitter()
atexit.register(_emitter.close)


def enabled() -> bool:
    return _emitter.enabled


def emit(kind: str, **fields: Any) -> None:
    _emitter.emit(kind, **fields)
