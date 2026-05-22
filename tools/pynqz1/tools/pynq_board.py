from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.bonsaid import DEFAULT_PORT  # noqa: E402
from tools import pynqctl  # noqa: E402


DEFAULT_BOARD_HOST = "pynq"
DEFAULT_REMOTE_DIR = "/home/xilinx/pynqz1-runtime"
DEFAULT_BOARD_PYTHON = "/usr/local/share/pynq-venv/bin/python"


@dataclass(frozen=True)
class BoardConfig:
    board_host: str
    ssh_user: str
    remote_dir: str
    board_python: str
    port: int
    heap_mib: int
    slab_mib: int
    overlay: str
    overlay_id: str
    rpc_host: str | None

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.board_host}"

    @property
    def runtime_host(self) -> str:
        return self.rpc_host or self.board_host


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rsync_runtime_command(config: BoardConfig, root: Path | None = None) -> list[str]:
    source = (root or project_root()) / "runtime"
    return [
        "rsync",
        "-a",
        str(source),
        f"{config.ssh_target}:{config.remote_dir}/",
    ]


def remote_daemon_command(config: BoardConfig) -> str:
    daemon = [
        "sudo",
        "env",
        "XILINX_XRT=/usr",
        "PYNQ_PYTHON=python3.10",
        config.board_python,
        "-m",
        "runtime.bonsaid",
        "--host",
        "0.0.0.0",
        "--port",
        str(config.port),
        "--allocator",
        "pynq",
        "--heap-mib",
        str(config.heap_mib),
        "--slab-mib",
        str(config.slab_mib),
        "--overlay",
        config.overlay,
        "--overlay-id",
        config.overlay_id,
    ]
    return f"cd {shlex.quote(config.remote_dir)} && exec {shlex.join(daemon)}"


def ssh_daemon_command(config: BoardConfig) -> list[str]:
    return ["ssh", "-tt", config.ssh_target, remote_daemon_command(config)]


def pynqctl_args(config: BoardConfig, command: list[str]) -> list[str]:
    return [
        "--host",
        config.runtime_host,
        "--port",
        str(config.port),
        *command,
    ]


def cmd_sync(args: argparse.Namespace) -> int:
    return run_command(rsync_runtime_command(config_from_args(args)))


def cmd_daemon(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    if not args.no_sync:
        rc = run_command(rsync_runtime_command(config))
        if rc != 0:
            return rc
    return run_command(ssh_daemon_command(config))


def cmd_hello(args: argparse.Namespace) -> int:
    return pynqctl.main(pynqctl_args(config_from_args(args), ["hello"]))


def cmd_smoke(args: argparse.Namespace) -> int:
    return pynqctl.main(
        pynqctl_args(
            config_from_args(args),
            ["smoke", "--bytes", args.nbytes],
        )
    )


def cmd_graph_copy_smoke(args: argparse.Namespace) -> int:
    return pynqctl.main(
        pynqctl_args(
            config_from_args(args),
            ["graph-copy-smoke", "--bytes", args.nbytes],
        )
    )


def run_command(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def config_from_args(args: argparse.Namespace) -> BoardConfig:
    return BoardConfig(
        board_host=args.board_host,
        ssh_user=args.ssh_user,
        remote_dir=args.remote_dir,
        board_python=args.board_python,
        port=args.port,
        heap_mib=args.heap_mib,
        slab_mib=args.slab_mib,
        overlay=args.overlay,
        overlay_id=args.overlay_id,
        rpc_host=args.rpc_host,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--board-host", default=DEFAULT_BOARD_HOST)
    parser.add_argument("--ssh-user", default="xilinx")
    parser.add_argument("--rpc-host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--board-python", default=DEFAULT_BOARD_PYTHON)
    parser.add_argument("--heap-mib", type=int, default=64)
    parser.add_argument("--slab-mib", type=int, default=32)
    parser.add_argument("--overlay", default="base")
    parser.add_argument("--overlay-id", default="pynq-base")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and exercise the Bonsai runtime on a PYNQ board"
    )
    add_common_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="copy runtime package to the board")
    sync.set_defaults(func=cmd_sync)

    daemon = subparsers.add_parser(
        "daemon",
        help="sync runtime and run board bonsaid in the foreground",
    )
    daemon.add_argument(
        "--no-sync",
        action="store_true",
        help="run the daemon from already-copied board files",
    )
    daemon.set_defaults(func=cmd_daemon)

    hello = subparsers.add_parser("hello", help="query board bonsaid over RPC")
    hello.set_defaults(func=cmd_hello)

    smoke = subparsers.add_parser("smoke", help="run board tensor transfer smoke")
    smoke.add_argument("--bytes", dest="nbytes", default="64k")
    smoke.set_defaults(func=cmd_smoke)

    graph_copy_smoke = subparsers.add_parser(
        "graph-copy-smoke",
        help="run board RUN_GRAPH COPY smoke",
    )
    graph_copy_smoke.add_argument("--bytes", dest="nbytes", default="64k")
    graph_copy_smoke.set_defaults(func=cmd_graph_copy_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except OSError as exc:
        print(f"pynq-board: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
