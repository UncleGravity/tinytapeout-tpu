from __future__ import annotations

import json

from tools import pynqctl


def cli_args(rpc_server, *args: str) -> list[str]:
    host, port = rpc_server.server_address
    return ["--host", host, "--port", str(port), *args]


def read_json(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_hello_prints_daemon_info(rpc_server, capsys):
    rc = pynqctl.main(cli_args(rpc_server, "hello"))

    assert rc == 0
    output = read_json(capsys)
    assert output["abi_version"] == 1
    assert output["server"] == "bonsaid"
    assert output["overlay_id"] == "test-overlay"


def test_smoke_round_trip(rpc_server, capsys):
    rc = pynqctl.main(cli_args(rpc_server, "smoke", "--bytes", "200k"))

    assert rc == 0
    output = read_json(capsys)
    assert output["ok"] is True
    assert output["nbytes"] == 200 * 1024
    assert output["upload"]["written"] == 200 * 1024
    assert output["download"]["read"] == 200 * 1024
    assert output["free"]["memory"]["tensor_count"] == 0


def test_graph_copy_smoke_round_trip(rpc_server, capsys):
    rc = pynqctl.main(cli_args(rpc_server, "graph-copy-smoke", "--bytes", "200k"))

    assert rc == 0
    output = read_json(capsys)
    assert output["ok"] is True
    assert output["nbytes"] == 200 * 1024
    assert output["graph"]["op_count"] == 1
    assert output["graph"]["outputs"] == [output["destination"]["tensor"]["handle"]]
    counters = dict(output["graph"]["counters"])
    elapsed_us = counters.pop("elapsed_us")
    assert elapsed_us >= 0
    assert counters == {
        "ps_ops": 1,
        "pl_ops": 0,
        "bytes_read": 200 * 1024,
        "bytes_written": 200 * 1024,
    }
    assert output["download"]["read"] == 200 * 1024
    assert output["free"][-1]["memory"]["tensor_count"] == 0


def test_file_upload_download_free(rpc_server, tmp_path, capsys):
    source = bytes((index * 29 + 11) % 251 for index in range(4096))
    src_path = tmp_path / "input.bin"
    dst_path = tmp_path / "output.bin"
    src_path.write_bytes(source)

    rc = pynqctl.main(cli_args(rpc_server, "alloc", "--bytes", str(len(source))))
    assert rc == 0
    handle = read_json(capsys)["tensor"]["handle"]

    rc = pynqctl.main(cli_args(rpc_server, "upload", str(handle), str(src_path)))
    assert rc == 0
    assert read_json(capsys)["written"] == len(source)

    rc = pynqctl.main(
        cli_args(
            rpc_server,
            "download",
            str(handle),
            str(dst_path),
            "--bytes",
            str(len(source)),
        )
    )
    assert rc == 0
    assert read_json(capsys)["read"] == len(source)
    assert dst_path.read_bytes() == source

    rc = pynqctl.main(cli_args(rpc_server, "free", str(handle)))
    assert rc == 0
    assert read_json(capsys)["memory"]["tensor_count"] == 0
