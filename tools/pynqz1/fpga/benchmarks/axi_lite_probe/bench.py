#!/usr/bin/env python3
"""AXI-Lite control-plane probe.

Exercises every protocol corner the W1A8 control path will rely on:
    - read RO constants (ID, VERSION) -> proves read decode + path
    - round-trip a scratch register   -> proves write -> readback
    - run/halt a free-running counter -> proves a write controls PL logic
    - measure counter rate            -> proves PL is actually clocked
    - reset-strobe a counter          -> proves write-side-effects (vs storage)

Failure of any one of these is loud enough that bringing up a real compute
kernel without running this first means debugging two layers at once.
"""

from __future__ import annotations

import argparse
import sys
import time

from pynq import Overlay

DEFAULT_BITFILE = "/home/xilinx/overlays/axi_lite_probe.bit"

REG_ID      = 0x00
REG_VERSION = 0x04
REG_SCRATCH = 0x08
REG_CTRL    = 0x0C
REG_COUNTER = 0x10

CTRL_RUN     = 1 << 0
CTRL_RESET   = 1 << 1   # write-only strobe; does not latch in ctrl_q

EXPECTED_ID      = 0xCAFE_0001
EXPECTED_VERSION = 0x0000_0001
EXPECTED_HZ      = 100_000_000   # FCLK_CLK0


def _check(name: str, got, want) -> int:
    if got == want:
        print(f"  ok   {name}: 0x{got:08x}")
        return 0
    print(f"  FAIL {name}: got 0x{got:08x}, want 0x{want:08x}", file=sys.stderr)
    return 1


def _check_pred(name: str, ok: bool, detail: str) -> int:
    if ok:
        print(f"  ok   {name}: {detail}")
        return 0
    print(f"  FAIL {name}: {detail}", file=sys.stderr)
    return 1


def probe(args: argparse.Namespace) -> int:
    overlay = Overlay(args.bitfile)
    regs = overlay.axi_lite_regs_0

    failures = 0

    print("[1] RO constants")
    failures += _check("ID",      regs.read(REG_ID),      EXPECTED_ID)
    failures += _check("VERSION", regs.read(REG_VERSION), EXPECTED_VERSION)

    print("[2] scratch round-trip")
    for pattern in (0xDEAD_BEEF, 0x1234_5678, 0xFFFF_FFFF, 0x0000_0000, 0xA5A5_5A5A):
        regs.write(REG_SCRATCH, pattern)
        got = regs.read(REG_SCRATCH)
        failures += _check(f"scratch=0x{pattern:08x}", got, pattern)

    print("[3] writes to RO are dropped (no fault, no state change)")
    regs.write(REG_SCRATCH, 0xCAFEBABE)
    regs.write(REG_ID,      0x11111111)   # should be ignored
    regs.write(REG_VERSION, 0x22222222)   # should be ignored
    failures += _check("ID untouched",      regs.read(REG_ID),      EXPECTED_ID)
    failures += _check("VERSION untouched", regs.read(REG_VERSION), EXPECTED_VERSION)
    failures += _check("scratch preserved", regs.read(REG_SCRATCH), 0xCAFEBABE)

    print("[4] counter halted while CTRL.run=0")
    regs.write(REG_CTRL, CTRL_RESET)      # clear + ensure run=0
    time.sleep(0.001)
    a = regs.read(REG_COUNTER)
    time.sleep(0.010)
    b = regs.read(REG_COUNTER)
    failures += _check_pred(
        "counter idle",
        a == b,
        f"before=0x{a:08x} after=0x{b:08x}",
    )

    print("[5] counter advances while CTRL.run=1")
    regs.write(REG_CTRL, CTRL_RESET | CTRL_RUN)   # clear then start
    a = regs.read(REG_COUNTER)
    sleep_s = 0.050
    time.sleep(sleep_s)
    b = regs.read(REG_COUNTER)
    delta = (b - a) & 0xFFFF_FFFF
    expected = int(EXPECTED_HZ * sleep_s)
    # ±25% to absorb sleep() jitter on the PS + the two AXI-Lite read latencies.
    lo, hi = int(expected * 0.75), int(expected * 1.25)
    failures += _check_pred(
        "counter rate",
        lo <= delta <= hi,
        f"delta={delta} ticks over {sleep_s*1000:.0f} ms "
        f"(want {lo}..{hi}, ~{EXPECTED_HZ/1e6:.0f} MHz)",
    )

    print("[6] reset strobe zeros the counter")
    regs.write(REG_CTRL, CTRL_RESET | CTRL_RUN)   # clear-while-running
    time.sleep(0.001)
    c = regs.read(REG_COUNTER)
    failures += _check_pred(
        "counter near-zero after strobe",
        c < EXPECTED_HZ // 100,   # < ~1 ms of accumulated ticks
        f"counter=0x{c:08x} ({c} ticks)",
    )

    print("[7] CTRL.run readback (only bit[0] latches; bit[1] is a strobe)")
    regs.write(REG_CTRL, CTRL_RUN)
    ctrl_read = regs.read(REG_CTRL)
    failures += _check_pred(
        "CTRL[0]=1, CTRL[1]=0 after write CTRL=0x1",
        ctrl_read == CTRL_RUN,
        f"got 0x{ctrl_read:08x}",
    )
    regs.write(REG_CTRL, 0)                       # stop
    ctrl_read = regs.read(REG_CTRL)
    failures += _check_pred(
        "CTRL=0 after write CTRL=0",
        ctrl_read == 0,
        f"got 0x{ctrl_read:08x}",
    )

    print()
    if failures:
        print(f"FAILED: {failures} check(s) failed", file=sys.stderr)
        return 1
    print("PASS: all AXI-Lite probes succeeded")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AXI-Lite control-plane probe")
    parser.add_argument("--bitfile", default=DEFAULT_BITFILE)
    parser.set_defaults(func=probe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
