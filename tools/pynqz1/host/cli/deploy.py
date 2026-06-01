"""Deploy + drive the daemon on a PYNQ-Z1 board.

One job: SSH + rsync. Three subcommands map 1:1 to remote actions:

  * ``sync``         rsync the board-side packages over to the board
  * ``build-native`` rebuild ``libbonsai_ps.so`` on the board via its Makefile
  * ``daemon``       start the daemon in the foreground

For RPC queries against a running daemon use ``pynqctl --host <board>``
directly (or set ``PYNQ_HOST=<board>``). This script no longer forwards
to pynqctl — chain them explicitly:

    pynq-deploy sync && pynq-deploy daemon &
    PYNQ_HOST=pynq pynqctl hello
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proto.ops import DEFAULT_PORT  # noqa: E402

DEFAULT_BOARD_HOST = "pynq"
DEFAULT_REMOTE_DIR = "/home/xilinx/pynqz1"
DEFAULT_BOARD_PYTHON = "/usr/local/share/pynq-venv/bin/python"

# Everything below board/ and proto/ gets rsync'd to the board. That's the
# entire on-board surface — the host backend and CLIs stay local.
BOARD_PACKAGES = ("board", "proto")

# Env vars on the host that get forwarded into the remote daemon process.
# PYNQ_PROFILE = "1" or path; affects board.profiling.events.
FORWARDED_ENV = ("PYNQ_PROFILE",)


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
    bitfile: str | None

    @property
    def ssh_target(self) -> str:
        return f"{self.ssh_user}@{self.board_host}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rsync_packages_command(config: BoardConfig, root: Path | None = None) -> list[str]:
    base = root or project_root()
    # --chmod=ug+w: nix-store sources are 0444/0555; without this the dest
    # tree on the board is un-writable and `make clean` fails.
    return [
        "rsync",
        "-a",
        "--chmod=ug+w",
        "--exclude=__pycache__",
        *[str(base / pkg) for pkg in BOARD_PACKAGES],
        f"{config.ssh_target}:{config.remote_dir}/",
    ]


def remote_daemon_command(config: BoardConfig) -> str:
    daemon = [
        "sudo",
        "env",
        "XILINX_XRT=/usr",
        "PYNQ_PYTHON=python3.10",
        f"PYTHONPATH={config.remote_dir}",
        *_forwarded_env(),
        config.board_python,
        "-m",
        "board.daemon",
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
    if config.bitfile is not None:
        daemon += ["--bitfile", config.bitfile]
    return f"cd {shlex.quote(config.remote_dir)} && exec {shlex.join(daemon)}"


def _forwarded_env() -> list[str]:
    forwarded: list[str] = []
    for name in FORWARDED_ENV:
        value = os.environ.get(name)
        if value is not None and value.lower() not in ("", "0", "false", "no", "off"):
            forwarded.append(f"{name}={value}")
    return forwarded


def remote_native_build_command(config: BoardConfig) -> str:
    # neon-fp16 (not neon-vfpv3) so the A9's hardware F16<->F32 conversion
    # (vcvt.f32.f16) is available to the flash-attention NEON intrinsics. The
    # A9 has no VFPv4/FMA, so do NOT use neon-vfpv4 here.
    # -mfp16-format=ieee is required on armv7 gcc for arm_neon.h to *declare*
    # the f16 intrinsics (vcvt_f32_f16 / vreinterpret_f16_u16 / float16x4_t);
    # without it they're implicitly-declared and the build fails. arm64/clang
    # declares them unconditionally, so the host test build doesn't need it.
    cflags = ("-O3 -mcpu=cortex-a9 -mfpu=neon-fp16 -mfloat-abi=hard "
              "-mfp16-format=ieee -std=c99 -fPIC")
    return (
        f"cd {shlex.quote(config.remote_dir)}/board/kernels/ps && "
        f"make OUT_DIR=. clean && CFLAGS={shlex.quote(cflags)} make OUT_DIR=."
    )


def ssh_daemon_command(config: BoardConfig) -> list[str]:
    return ["ssh", "-tt", config.ssh_target, remote_daemon_command(config)]


def ssh_native_build_command(config: BoardConfig) -> list[str]:
    return ["ssh", "-tt", config.ssh_target, remote_native_build_command(config)]


def cmd_sync(args: argparse.Namespace) -> int:
    return run_command(rsync_packages_command(config_from_args(args)))


def cmd_daemon(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    if not args.no_sync:
        rc = run_command(rsync_packages_command(config))
        if rc != 0:
            return rc
    return run_command(ssh_daemon_command(config))


def cmd_build_native(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    if not args.no_sync:
        rc = run_command(rsync_packages_command(config))
        if rc != 0:
            return rc
    return run_command(ssh_native_build_command(config))


def run_command(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def config_from_args(args: argparse.Namespace) -> BoardConfig:
    # Daemon-only fields are absent on sync/build-native; defaults are unused
    # there but BoardConfig is frozen so we have to populate them.
    return BoardConfig(
        board_host=args.board_host,
        ssh_user=args.ssh_user,
        remote_dir=args.remote_dir,
        board_python=args.board_python,
        port=args.port,
        heap_mib=getattr(args, "heap_mib", 0),
        slab_mib=getattr(args, "slab_mib", 0),
        overlay=getattr(args, "overlay", ""),
        overlay_id=getattr(args, "overlay_id", ""),
        bitfile=getattr(args, "bitfile", None),
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--board-host", default=os.environ.get("PYNQ_HOST", DEFAULT_BOARD_HOST))
    parser.add_argument("--ssh-user", default="xilinx")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PYNQ_PORT", DEFAULT_PORT)))
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--board-python", default=DEFAULT_BOARD_PYTHON)


def add_daemon_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--heap-mib", type=int, default=64)
    parser.add_argument("--slab-mib", type=int, default=32)
    parser.add_argument("--overlay", default="base")
    parser.add_argument("--overlay-id", default="pynq-base")
    parser.add_argument(
        "--bitfile",
        default=None,
        help="path to a PL bitstream ON THE BOARD; forwarded to board.daemon "
             "to load the overlay and register PL kernels",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and exercise the daemon on a PYNQ board"
    )
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="rsync board/ and proto/ to the board")
    sync.set_defaults(func=cmd_sync)

    daemon = subparsers.add_parser(
        "daemon", help="sync packages and run the board daemon in the foreground"
    )
    daemon.add_argument(
        "--no-sync", action="store_true",
        help="run from already-copied board files",
    )
    add_daemon_args(daemon)
    daemon.set_defaults(func=cmd_daemon)

    build_native = subparsers.add_parser(
        "build-native", help="sync packages and rebuild libbonsai_ps.so on the board"
    )
    build_native.add_argument(
        "--no-sync", action="store_true",
        help="build from already-copied board files",
    )
    build_native.set_defaults(func=cmd_build_native)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except OSError as exc:
        print(f"pynq-deploy: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
