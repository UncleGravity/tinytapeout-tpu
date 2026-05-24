"""Tests for the pynq-profile analyzer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_profile_module():
    """``host/cli/pynq-profile.py`` uses a hyphen, so import it by path."""
    path = Path(__file__).resolve().parents[1] / "cli" / "pynq-profile.py"
    spec = importlib.util.spec_from_file_location("pynq_profile", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pynq_profile"] = module
    spec.loader.exec_module(module)
    return module


pynq_profile = _load_profile_module()


def write_events(path: Path, events: list[dict]) -> None:
    with path.open("w") as fp:
        for ev in events:
            fp.write(json.dumps(ev) + "\n")


def test_correlate_pairs_host_and_board(tmp_path):
    path = tmp_path / "ev.ndjson"
    write_events(path, [
        {"t_us": 1000, "side": "host", "kind": "rpc_send", "req_id": 7, "op": "RUN_GRAPH", "bytes": 512},
        {"t_us": 1100, "side": "board", "kind": "graph_begin", "req_id": 7, "op_count": 1},
        {"t_us": 1150, "side": "board", "kind": "op_end", "req_id": 7, "op": "MATMUL_Q1A8",
         "index": 0, "total_us": 50, "read_us": 5, "compute_us": 40, "write_us": 5,
         "bytes_read": 1024, "bytes_written": 32},
        {"t_us": 1200, "side": "board", "kind": "graph_end", "req_id": 7,
         "elapsed_us": 50, "bytes_read": 1024, "bytes_written": 32, "ps_ops": 1, "pl_ops": 0},
        {"t_us": 1300, "side": "host", "kind": "rpc_recv", "req_id": 7, "op": "RUN_GRAPH", "us": 300, "bytes": 128},
    ])

    calls = pynq_profile.correlate(pynq_profile.iter_events([path]))
    assert set(calls.keys()) == {7}
    call = calls[7]
    assert call.op == "RUN_GRAPH"
    assert call.host_send_us == 1000
    assert call.host_recv_us == 1300
    assert call.round_trip_us == 300
    assert call.board_graph_us == 50
    assert call.rpc_overhead_us == 250
    assert len(call.op_spans) == 1


def test_summary_runs_clean(tmp_path, capsys):
    path = tmp_path / "ev.ndjson"
    write_events(path, [
        {"t_us": 0, "side": "host", "kind": "rpc_send", "req_id": 1, "op": "RUN_GRAPH", "bytes": 100},
        {"t_us": 50, "side": "board", "kind": "op_end", "req_id": 1, "op": "ADD_F32",
         "index": 0, "total_us": 40, "read_us": 5, "compute_us": 30, "write_us": 5,
         "bytes_read": 64, "bytes_written": 32},
        {"t_us": 60, "side": "board", "kind": "graph_end", "req_id": 1,
         "elapsed_us": 45, "bytes_read": 64, "bytes_written": 32, "ps_ops": 1, "pl_ops": 0},
        {"t_us": 100, "side": "host", "kind": "rpc_recv", "req_id": 1, "op": "RUN_GRAPH", "us": 100, "bytes": 32},
    ])

    rc = pynq_profile.main(["summary", str(path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "pynq profile" in captured.out
    assert "ADD_F32" in captured.out


def test_speedscope_emits_valid_json(tmp_path, capsys):
    path = tmp_path / "ev.ndjson"
    write_events(path, [
        {"t_us": 0, "side": "host", "kind": "rpc_send", "req_id": 1, "op": "RUN_GRAPH", "bytes": 100},
        {"t_us": 50, "side": "board", "kind": "op_end", "req_id": 1, "op": "COPY",
         "index": 0, "total_us": 25, "read_us": 10, "write_us": 10, "bytes_read": 64, "bytes_written": 64},
        {"t_us": 100, "side": "host", "kind": "rpc_recv", "req_id": 1, "op": "RUN_GRAPH", "us": 100, "bytes": 32},
    ])
    rc = pynq_profile.main(["speedscope", str(path)])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["exporter"] == "pynq-profile"
    assert any(frame["name"] == "COPY" for frame in payload["shared"]["frames"])


def test_handles_empty_input(tmp_path, capsys):
    path = tmp_path / "empty.ndjson"
    path.write_text("")
    rc = pynq_profile.main(["summary", str(path)])
    assert rc == 1
