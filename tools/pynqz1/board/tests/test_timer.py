from __future__ import annotations

import time

from board.profiling.timer import Timer


def test_op_records_total_us():
    timer = Timer(profile=True)
    with timer.op("FOO", index=0):
        time.sleep(0.001)
    assert len(timer.ops) == 1
    assert timer.ops[0].op == "FOO"
    assert timer.ops[0].total_us > 0


def test_section_accumulates_under_current_op():
    timer = Timer(profile=True)
    with timer.op("BAR", index=0):
        with timer.section("read"):
            time.sleep(0.0005)
        with timer.section("read"):
            time.sleep(0.0005)
    span = timer.ops[0]
    assert span.sections["read_us"] > 0


def test_add_accumulates_fields_per_op():
    timer = Timer(profile=True)
    with timer.op("MATMUL", index=0):
        timer.add("bytes_read", 128)
        timer.add("bytes_read", 256)
    assert timer.ops[0].fields["bytes_read"] == 384


def test_summary_shape():
    timer = Timer(profile=True)
    with timer.op("FOO", index=0):
        timer.add("bytes_read", 16)
    payload = timer.summary(counters={"ps_ops": 1})
    assert payload["counters"]["ps_ops"] == 1
    assert payload["ops"][0]["op"] == "FOO"
    assert payload["ops"][0]["bytes_read"] == 16


def test_disabled_skips_spans_but_tracks_byte_totals():
    # The production path (profiling off): op/section are no-ops and no per-op
    # spans are built, but the two byte counters the RUN_GRAPH response needs
    # are still accumulated.
    timer = Timer(profile=False)
    with timer.op("FOO"):
        with timer.section("read"):
            timer.add("bytes_read", 100)
        timer.add("bytes_written", 40)
        timer.add("rows", 8)  # non-byte field is simply ignored when disabled
    assert timer.ops == []
    assert timer.total_bytes_read == 100
    assert timer.total_bytes_written == 40
