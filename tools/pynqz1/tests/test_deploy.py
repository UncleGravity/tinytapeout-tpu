"""Argv-builder tests for host.cli.deploy. No subprocess execution."""

from __future__ import annotations

from pathlib import Path

from host.cli import deploy


def config(**overrides):
    values = {
        "board_host": "pynq",
        "ssh_user": "xilinx",
        "remote_dir": "/home/xilinx/pynqz1",
        "board_python": "/usr/local/share/pynq-venv/bin/python",
        "port": 50055,
        "heap_mib": 64,
        "slab_mib": 32,
        "overlay": "base",
        "overlay_id": "pynq-base",
    }
    values.update(overrides)
    return deploy.BoardConfig(**values)


def test_rsync_includes_every_board_package():
    command = deploy.rsync_packages_command(config(), Path("/repo/pynqz1"))

    assert command[0] == "rsync"
    assert "-a" in command
    assert "--chmod=ug+w" in command
    assert "--exclude=__pycache__" in command
    sources = [arg for arg in command if arg.startswith("/repo/pynqz1/")]
    assert sources == ["/repo/pynqz1/board", "/repo/pynqz1/proto"]
    assert command[-1] == "xilinx@pynq:/home/xilinx/pynqz1/"


def test_ssh_daemon_runs_board_daemon_module():
    command = deploy.ssh_daemon_command(
        config(board_host="pynq.local", port=50060, heap_mib=4, slab_mib=1)
    )

    assert command[:3] == ["ssh", "-tt", "xilinx@pynq.local"]
    remote = command[3]
    assert "python -m board.daemon" in remote
    assert "--port 50060 --allocator pynq" in remote
    assert "--heap-mib 4 --slab-mib 1" in remote
    assert "PYNQ_PS_LIB" not in remote


def test_ssh_daemon_forwards_profile_env(monkeypatch):
    monkeypatch.setenv("PYNQ_PROFILE", "1")
    command = deploy.ssh_daemon_command(config())
    assert "PYNQ_PROFILE=1" in command[3]


def test_ssh_daemon_forwards_profile_path(monkeypatch):
    monkeypatch.setenv("PYNQ_PROFILE", "/var/log/pynq.ndjson")
    command = deploy.ssh_daemon_command(config())
    assert "PYNQ_PROFILE=/var/log/pynq.ndjson" in command[3]


def test_ssh_native_build_invokes_makefile():
    command = deploy.ssh_native_build_command(config())
    remote = command[3]
    assert "board/kernels/ps" in remote
    assert "make" in remote
    assert "-mcpu=cortex-a9" in remote


def test_daemon_syncs_before_ssh(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(deploy, "run_command", lambda c: commands.append(c) or 0)

    rc = deploy.main(["daemon"])

    assert rc == 0
    assert commands[0][0] == "rsync"
    assert commands[1][:3] == ["ssh", "-tt", "xilinx@pynq"]


def test_build_native_syncs_before_ssh(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(deploy, "run_command", lambda c: commands.append(c) or 0)

    rc = deploy.main(["build-native"])

    assert rc == 0
    assert commands[0][0] == "rsync"
    assert commands[1][:3] == ["ssh", "-tt", "xilinx@pynq"]


def test_no_sync_skips_rsync(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(deploy, "run_command", lambda c: commands.append(c) or 0)
    rc = deploy.main(["daemon", "--no-sync"])
    assert rc == 0
    assert len(commands) == 1
    assert commands[0][:3] == ["ssh", "-tt", "xilinx@pynq"]


def test_board_host_from_env(monkeypatch):
    monkeypatch.setenv("PYNQ_HOST", "pynq.local")
    commands: list[list[str]] = []
    monkeypatch.setattr(deploy, "run_command", lambda c: commands.append(c) or 0)
    deploy.main(["daemon", "--no-sync"])
    assert "xilinx@pynq.local" in commands[0]
