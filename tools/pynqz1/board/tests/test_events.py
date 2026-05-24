"""Verify board.profiling.events writes well-formed NDJSON to file or stderr."""

from __future__ import annotations

import importlib
import json

import pytest


@pytest.fixture
def reload_events(monkeypatch):
    """Re-import events with a controlled PYNQ_PROFILE value each test."""
    def _reload(value: str | None):
        if value is None:
            monkeypatch.delenv("PYNQ_PROFILE", raising=False)
        else:
            monkeypatch.setenv("PYNQ_PROFILE", value)
        import board.profiling.events as events_module
        importlib.reload(events_module)
        return events_module
    return _reload


def test_disabled_by_default(reload_events):
    events = reload_events(None)
    assert not events.enabled()
    events.emit("test", x=1)  # must not raise


def test_emit_to_file(reload_events, tmp_path):
    path = tmp_path / "events.ndjson"
    events = reload_events(str(path))
    events.emit("graph_begin", req_id=42, op_count=3)
    events.emit("op_end", req_id=42, op="MATMUL_Q1A8", index=0, total_us=123)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["side"] == "board"
    assert rec0["kind"] == "graph_begin"
    assert rec0["req_id"] == 42
    assert rec0["op_count"] == 3
    assert "t_us" in rec0

    rec1 = json.loads(lines[1])
    assert rec1["kind"] == "op_end"
    assert rec1["total_us"] == 123


def test_emit_to_stderr(reload_events, capsys):
    events = reload_events("1")
    events.emit("hello", req_id=1)
    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert "\"req_id\":1" in captured.err


def test_falsy_values_disable(reload_events):
    for value in ("0", "false", "no", "off", ""):
        events = reload_events(value)
        assert not events.enabled(), f"value={value!r} should disable"
