from __future__ import annotations

from pathlib import Path

from tools import pynq_board


def config(**overrides):
    values = {
        "board_host": "pynq",
        "ssh_user": "xilinx",
        "remote_dir": "/home/xilinx/pynqz1-runtime",
        "board_python": "/usr/local/share/pynq-venv/bin/python",
        "port": 50055,
        "heap_mib": 64,
        "slab_mib": 32,
        "overlay": "base",
        "overlay_id": "pynq-base",
        "rpc_host": None,
    }
    values.update(overrides)
    return pynq_board.BoardConfig(**values)


def test_rsync_runtime_command_uses_runtime_package_path():
    command = pynq_board.rsync_runtime_command(config(), Path("/repo/pynqz1"))

    assert command == [
        "rsync",
        "-a",
        "/repo/pynqz1/runtime",
        "xilinx@pynq:/home/xilinx/pynqz1-runtime/",
    ]


def test_ssh_daemon_command_uses_board_python_overlay_and_heap():
    command = pynq_board.ssh_daemon_command(
        config(
            board_host="pynq.local",
            remote_dir="/home/xilinx/runtime scratch",
            port=50060,
            heap_mib=4,
            slab_mib=1,
            overlay_id="graph-copy",
        )
    )

    assert command[:3] == ["ssh", "-tt", "xilinx@pynq.local"]
    remote = command[3]
    assert remote.startswith("cd '/home/xilinx/runtime scratch' && exec sudo env ")
    assert "PYNQ_PS_LIB=/home/xilinx/runtime scratch/runtime/native/libbonsai_ps.so" in remote
    assert "/usr/local/share/pynq-venv/bin/python -m runtime.bonsaid" in remote
    assert "--host 0.0.0.0 --port 50060 --allocator pynq" in remote
    assert "--heap-mib 4 --slab-mib 1 --overlay base --overlay-id graph-copy" in remote


def test_ssh_daemon_command_forwards_profile_env(monkeypatch):
    monkeypatch.setenv("PYNQ_PROFILE", "1")

    command = pynq_board.ssh_daemon_command(config())

    assert "PYNQ_PROFILE=1" in command[3]


def test_ssh_native_build_command_compiles_runtime_library():
    command = pynq_board.ssh_native_build_command(config(board_host="pynq.local"))

    assert command[:3] == ["ssh", "-tt", "xilinx@pynq.local"]
    remote = command[3]
    assert remote.startswith("cd /home/xilinx/pynqz1-runtime && ")
    assert "gcc -O3 -mcpu=cortex-a9 -mfpu=neon-vfpv3 -mfloat-abi=hard" in remote
    assert "-std=c99 -fPIC -shared" in remote
    assert "-o /tmp/libbonsai_ps.so runtime/native/bonsai_ps.c -lm" in remote
    assert "sudo install -m 0755 /tmp/libbonsai_ps.so runtime/native/libbonsai_ps.so" in remote


def test_daemon_syncs_before_ssh(monkeypatch):
    commands = []

    def fake_run(command):
        commands.append(command)
        return 0

    monkeypatch.setattr(pynq_board, "run_command", fake_run)

    rc = pynq_board.main(["--heap-mib", "4", "--slab-mib", "1", "daemon"])

    assert rc == 0
    assert commands[0][0] == "rsync"
    assert commands[1][:3] == ["ssh", "-tt", "xilinx@pynq"]
    assert "--heap-mib 4 --slab-mib 1" in commands[1][3]


def test_build_native_syncs_before_ssh(monkeypatch):
    commands = []

    def fake_run(command):
        commands.append(command)
        return 0

    monkeypatch.setattr(pynq_board, "run_command", fake_run)

    rc = pynq_board.main(["build-native"])

    assert rc == 0
    assert commands[0][0] == "rsync"
    assert commands[1][:3] == ["ssh", "-tt", "xilinx@pynq"]
    assert "runtime/native/libbonsai_ps.so" in commands[1][3]


def test_graph_copy_smoke_uses_rpc_host(monkeypatch):
    calls = []

    def fake_pynqctl_main(argv):
        calls.append(argv)
        return 17

    monkeypatch.setattr(pynq_board.pynqctl, "main", fake_pynqctl_main)

    rc = pynq_board.main(
        [
            "--rpc-host",
            "pynq-rpc",
            "--port",
            "50060",
            "graph-copy-smoke",
            "--bytes",
            "1536k",
        ]
    )

    assert rc == 17
    assert calls == [
        [
            "--host",
            "pynq-rpc",
            "--port",
            "50060",
            "graph-copy-smoke",
            "--bytes",
            "1536k",
        ]
    ]
