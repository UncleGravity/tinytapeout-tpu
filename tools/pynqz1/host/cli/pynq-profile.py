"""Aggregate PYNQ_PROFILE event logs into a CLI summary or speedscope JSON.

Both the host backend and the board daemon emit one NDJSON event per
significant action when ``PYNQ_PROFILE`` is set:

    PYNQ_PROFILE=/tmp/host.ndjson  nix run .#llama -- -m model.gguf ...
    # on the board (set when starting the daemon):
    PYNQ_PROFILE=/var/log/board.ndjson  python -m board.daemon ...

Then:

    pynq-profile summary   /tmp/host.ndjson /var/log/board.ndjson
    pynq-profile speedscope /tmp/host.ndjson /var/log/board.ndjson > flame.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# -- ingest --------------------------------------------------------------


def iter_events(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open() as fp:
            for line_no, raw in enumerate(fp, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(
                        f"warning: skipping {path}:{line_no}: {exc}",
                        file=sys.stderr,
                    )


# -- correlation ---------------------------------------------------------


@dataclass
class RpcCall:
    req_id: int
    op: str
    host_send_us: int | None = None
    host_recv_us: int | None = None
    host_send_bytes: int = 0
    host_recv_bytes: int = 0
    board_graph_us: int | None = None  # from graph_end counter
    board_bytes_read: int = 0
    board_bytes_written: int = 0
    op_spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def round_trip_us(self) -> int | None:
        if self.host_send_us is None or self.host_recv_us is None:
            return None
        return self.host_recv_us - self.host_send_us

    @property
    def rpc_overhead_us(self) -> int | None:
        rt = self.round_trip_us
        if rt is None or self.board_graph_us is None:
            return None
        return max(0, rt - self.board_graph_us)


def correlate(events: Iterable[dict[str, Any]]) -> dict[int, RpcCall]:
    calls: dict[int, RpcCall] = {}

    def get(req_id: int, op_hint: str = "?") -> RpcCall:
        if req_id not in calls:
            calls[req_id] = RpcCall(req_id=req_id, op=op_hint)
        if op_hint != "?" and calls[req_id].op == "?":
            calls[req_id].op = op_hint
        return calls[req_id]

    for ev in events:
        kind = ev.get("kind")
        req_id = ev.get("req_id")
        if req_id is None:
            continue
        t_us = int(ev.get("t_us", 0))

        if kind == "rpc_send":
            call = get(req_id, ev.get("op", "?"))
            call.host_send_us = t_us
            call.host_send_bytes = int(ev.get("bytes", 0))
        elif kind == "rpc_recv":
            call = get(req_id)
            call.host_recv_us = t_us
            call.host_recv_bytes = int(ev.get("bytes", 0))
        elif kind == "graph_end":
            call = get(req_id)
            call.board_graph_us = int(ev.get("elapsed_us", 0))
            call.board_bytes_read = int(ev.get("bytes_read", 0))
            call.board_bytes_written = int(ev.get("bytes_written", 0))
        elif kind == "op_end":
            call = get(req_id)
            call.op_spans.append(ev)

    return calls


# -- summary -------------------------------------------------------------


def _fmt_time(us: int) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:7.2f}s"
    if us >= 1_000:
        return f"{us / 1_000:7.2f}ms"
    return f"{us:5d}µs"


def _bar(fraction: float, width: int = 40) -> str:
    # Clamp so a malformed section (e.g. a field that snuck through the
    # ``*_us`` filter holding a wall-clock timestamp) cannot OOM the tool
    # trying to allocate billions of bar characters.
    clamped = max(0.0, min(1.0, fraction))
    return "█" * int(round(clamped * width))


def cmd_summary(args: argparse.Namespace) -> int:
    calls = correlate(iter_events(args.files))
    if not calls:
        print("no events", file=sys.stderr)
        return 1

    total_rt_us = sum(c.round_trip_us or 0 for c in calls.values())
    total_rpc_us = sum(c.rpc_overhead_us or 0 for c in calls.values())

    op_totals: dict[str, int] = defaultdict(int)
    op_calls: dict[str, int] = defaultdict(int)
    op_durations: dict[str, list[int]] = defaultdict(list)
    op_bytes: dict[str, int] = defaultdict(int)
    # Per-op, per-section totals. Any span field ending in ``_us`` except
    # ``total_us`` is treated as a timer section emitted by the kernel.
    op_sections: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for call in calls.values():
        for span in call.op_spans:
            name = str(span.get("op", "?"))
            us = int(span.get("total_us", 0))
            op_totals[name] += us
            op_calls[name] += 1
            op_durations[name].append(us)
            op_bytes[name] += int(span.get("bytes_read", 0)) + int(span.get("bytes_written", 0))
            for key, value in span.items():
                # ``total_us`` is the op wall time, not a section.
                # ``t_us`` is the event timestamp injected by events.py.
                if key in ("total_us", "t_us") or not key.endswith("_us"):
                    continue
                try:
                    op_sections[name][key[:-3]] += int(value)
                except (TypeError, ValueError):
                    continue

    total_compute_us = sum(int(s.get("compute_us", 0)) for c in calls.values() for s in c.op_spans)
    total_memory_us = sum(
        int(s.get("read_us", 0)) + int(s.get("write_us", 0))
        for c in calls.values() for s in c.op_spans
    )

    print()
    print(f"=== pynq profile  ({len(calls)} RPC calls) ===")
    print()
    print(f"Total wall (round-trip sum)   {_fmt_time(total_rt_us)}")
    print(f"Board kernel compute          {_fmt_time(total_compute_us)}")
    print(f"Board memory r/w              {_fmt_time(total_memory_us)}")
    print(f"RPC overhead (rt − board)     {_fmt_time(total_rpc_us)}")
    print()
    if total_rt_us > 0:
        print("Categories:")
        for label, value in (
            ("Math (compute)", total_compute_us),
            ("Memory (r/w)  ", total_memory_us),
            ("RPC overhead  ", total_rpc_us),
        ):
            frac = value / total_rt_us
            print(f"  {label}  {_fmt_time(value)}  {frac * 100:5.1f}%  {_bar(frac)}")
        print()

    # Per top-level RPC (HELLO / ALLOC / UPLOAD / DOWNLOAD / RUN_GRAPH / FREE).
    # The compute/memory/rpc breakdown above is derived from RUN_GRAPH op-spans
    # only, so tensor-transfer RPCs are invisible there. This table accounts
    # for the *whole* round-trip wall so upload/download costs are visible.
    rpc_totals: dict[str, int] = defaultdict(int)
    rpc_calls: dict[str, int] = defaultdict(int)
    rpc_bytes: dict[str, int] = defaultdict(int)
    for call in calls.values():
        rt = call.round_trip_us
        if rt is None:
            continue
        rpc_totals[call.op] += rt
        rpc_calls[call.op] += 1
        rpc_bytes[call.op] += call.host_send_bytes + call.host_recv_bytes
    if rpc_totals:
        print("By RPC (top-level round-trip):")
        print(f"  {'op':<20} {'calls':>6} {'total':>10} {'avg':>10} {'bytes':>12}")
        for name, total in sorted(rpc_totals.items(), key=lambda kv: kv[1], reverse=True):
            count = rpc_calls[name]
            avg = total // count if count else 0
            frac = total / total_rt_us if total_rt_us else 0.0
            print(
                f"  {name:<20} {count:>6} {_fmt_time(total):>10} "
                f"{_fmt_time(avg):>10} {rpc_bytes[name]:>12}  {frac * 100:5.1f}%"
            )
        print()

    if op_totals:
        print("By op:")
        print(f"  {'op':<18} {'calls':>6} {'total':>10} {'avg':>10} {'p99':>10} {'bytes':>12}")
        ordered = sorted(op_totals.items(), key=lambda kv: kv[1], reverse=True)
        for name, total in ordered:
            durs = sorted(op_durations[name])
            count = op_calls[name]
            avg = total // count if count else 0
            p99 = durs[max(0, int(0.99 * len(durs)) - 1)] if durs else 0
            print(
                f"  {name:<18} {count:>6} {_fmt_time(total):>10} "
                f"{_fmt_time(avg):>10} {_fmt_time(p99):>10} {op_bytes[name]:>12}"
            )
        print()

        # Per-op section breakdown. Each kernel emits its own timer.section()
        # calls; we surface them here so the dominant sub-step is visible
        # without grepping the raw ndjson.
        ops_with_sections = [
            (name, total) for name, total in ordered
            if op_sections.get(name) and len(op_sections[name]) > 1
        ]
        if ops_with_sections:
            print("Section breakdown:")
            for name, total in ops_with_sections:
                sections = sorted(
                    op_sections[name].items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                print(f"  {name}  ({op_calls[name]} calls, {_fmt_time(total)} total)")
                for section, value in sections:
                    frac = value / total if total > 0 else 0.0
                    print(
                        f"    {section:<16} {_fmt_time(value):>10}  "
                        f"{frac * 100:5.1f}%  {_bar(frac, width=30)}"
                    )
                print()
    return 0


# -- speedscope ----------------------------------------------------------


def cmd_speedscope(args: argparse.Namespace) -> int:
    """Emit a flat speedscope profile (https://www.speedscope.app/)."""
    calls = correlate(iter_events(args.files))
    name_idx: dict[str, int] = {}
    shared: list[dict[str, str]] = []

    def frame_id(name: str) -> int:
        if name not in name_idx:
            name_idx[name] = len(shared)
            shared.append({"name": name})
        return name_idx[name]

    spans: list[tuple[int, int, int]] = []  # (start_us, end_us, frame_id)
    for call in sorted(calls.values(), key=lambda c: c.host_send_us or 0):
        for span in call.op_spans:
            dur = int(span.get("total_us", 0))
            if dur <= 0:
                continue
            # No absolute start for op_end; place sequentially.
            # For nicer flame view, group under the call's RPC span.
            spans.append((dur, frame_id(span.get("op", "?")), call.req_id))

    cursor = 0
    flat_events: list[dict[str, Any]] = []
    for dur, frame, _req in spans:
        flat_events.append({"type": "O", "at": cursor, "frame": frame})
        cursor += dur
        flat_events.append({"type": "C", "at": cursor, "frame": frame})

    profile = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "exporter": "pynq-profile",
        "name": "pynq inference",
        "activeProfileIndex": 0,
        "shared": {"frames": shared},
        "profiles": [{
            "type": "evented",
            "name": "ops",
            "unit": "microseconds",
            "startValue": 0,
            "endValue": cursor,
            "events": flat_events,
        }],
    }
    json.dump(profile, sys.stdout)
    return 0


# -- entry --------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate PYNQ_PROFILE event logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary = subparsers.add_parser("summary", help="terminal-friendly aggregate (default)")
    summary.add_argument("files", nargs="+", type=Path)
    summary.set_defaults(func=cmd_summary)

    speed = subparsers.add_parser("speedscope", help="emit speedscope JSON to stdout")
    speed.add_argument("files", nargs="+", type=Path)
    speed.set_defaults(func=cmd_speedscope)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
