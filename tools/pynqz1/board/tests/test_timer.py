from __future__ import annotations

import json
import os
import time

from board.profiling.timer import Timer


def test_op_records_total_us():
    timer = Timer()
    with timer.op("FOO", index=0):
        time.sleep(0.001)
    assert len(timer.ops) == 1
    assert timer.ops[0].op == "FOO"
    assert timer.ops[0].total_us > 0


def test_section_accumulates_under_current_op():
    timer = Timer()
    with timer.op("BAR", index=0):
        with timer.section("read"):
            time.sleep(0.0005)
        with timer.section("read"):
            time.sleep(0.0005)
    span = timer.ops[0]
    assert span.sections["read_us"] > 0


def test_add_accumulates_fields_per_op():
    timer = Timer()
    with timer.op("MATMUL", index=0):
        timer.add("bytes_read", 128)
        timer.add("bytes_read", 256)
    assert timer.ops[0].fields["bytes_read"] == 384


def test_emit_writes_json_only_when_enabled(capsys, monkeypatch):
    timer = Timer()
    with timer.op("FOO", index=0):
        pass

    monkeypatch.delenv("PYNQ_PROFILE", raising=False)
    timer.emit_if_enabled()
    assert capsys.readouterr().err == ""

    monkeypatch.setenv("PYNQ_PROFILE", "1")
    timer.emit_if_enabled(counters={"ps_ops": 1})
    captured = capsys.readouterr()
    assert captured.err.startswith("pynq profile: ")
    payload = json.loads(captured.err[len("pynq profile: ") :])
    assert payload["counters"]["ps_ops"] == 1
    assert payload["ops"][0]["op"] == "FOO"
