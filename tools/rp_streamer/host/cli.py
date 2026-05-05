"""
Host CLI for the rp_streamer firmware.

Subcommands:
  smoke            run a few hand-crafted matmul cases against the chip and check psums
  debug            single-case run with intermediate RDP sweeps to isolate which phase
                   (SEED / START / RDP) introduced an error
  flash-bitstream  upload an iCE40 bitstream into RP2350 flash and reload the FPGA

Transport: vendor-class USB bulk endpoints over libusb (via pyusb). The
firmware does not expose a CDC tty; device discovery is by VID:PID.

Usage:
  python host/cli.py smoke
  python host/cli.py debug
  python host/cli.py flash-bitstream <bitstream.bin>
"""
import argparse
import pathlib
import struct

from protocol import (
    CMD_CLEAR, CMD_LDA, CMD_LDW, CMD_NOP, CMD_RDP, CMD_SEED, CMD_START, CMD_STATUS,
    S_ALL_VALID, S_DONE, S_ERROR, S_START_READY,
    ROWS, COLS, PSUM_BYTES,
    open_device, pack_ui, x_frame, reset_chip, to_signed16,
)


def setup_frame(w, a, seeds):
    """CLEAR + LDW (per row) + LDA (per col) + SEED (per row,byte) + STATUS."""
    pairs = [(pack_ui(CMD_CLEAR), 0), (pack_ui(CMD_NOP), 0)]
    for r in range(ROWS):
        packed = sum((w[r][c] & 1) << c for c in range(COLS))
        pairs.append((pack_ui(CMD_LDW, arg=r), packed))
    for c in range(COLS):
        pairs.append((pack_ui(CMD_LDA, arg=c), a[c] & 0xFF))
    for r in range(ROWS):
        sraw = seeds[r] & 0xFFFF
        for b in range(PSUM_BYTES):
            pairs.append((
                pack_ui(CMD_SEED, arg=(r | (b << 1))),
                (sraw >> (8 * b)) & 0xFF,
            ))
    pairs.append((pack_ui(CMD_STATUS), 0))
    return pairs


def expected_psums(w, a, seeds):
    out = []
    for r in range(ROWS):
        v = seeds[r]
        for c in range(COLS):
            v += a[c] if w[r][c] else -a[c]
        out.append(v)
    return out


def run_one(conn, w, a, seeds, polls=16):
    """Run setup + START + poll + RDP all in two X-frames; return (psums, err)."""
    reset_chip(conn)

    setup_resp = x_frame(conn, setup_frame(w, a, seeds))
    pre = setup_resp[-1]
    if pre & S_ERROR:
        return None, f"pre-START ERROR set, status=0x{pre:02x}"
    if not (pre & S_START_READY):
        return None, f"pre-START missing START_READY, status=0x{pre:02x}"

    runframe = [(pack_ui(CMD_START), 0), (pack_ui(CMD_NOP), 0)]
    runframe += [(pack_ui(CMD_STATUS), 0)] * polls
    for r in range(ROWS):
        for b in range(PSUM_BYTES):
            runframe.append((pack_ui(CMD_RDP, arg=(r | (b << 1))), 0))

    resp = x_frame(conn, runframe)
    poll_resp = resp[2:2 + polls]
    rdp_resp = resp[2 + polls:]

    if not any(st & S_DONE for st in poll_resp):
        return None, f"no DONE in {polls} polls; last=0x{poll_resp[-1]:02x}"
    last_poll = poll_resp[-1]
    if last_poll & S_ERROR:
        return None, f"ERROR in poll, status=0x{last_poll:02x}"
    if not (last_poll & S_ALL_VALID):
        return None, f"DONE without ALL_VALID, status=0x{last_poll:02x}"

    psums = []
    for r in range(ROWS):
        raw = 0
        for b in range(PSUM_BYTES):
            raw |= rdp_resp[r * PSUM_BYTES + b] << (8 * b)
        psums.append(to_signed16(raw))
    return psums, None


# ----------------------------------------------------------------------------
# subcommand: smoke

SMOKE_CASES = [
    ([[1, 1], [1, 1]], [10, 20], [0, 0]),         # -> [30, 30]
    ([[1, 0], [1, 1]], [3, 5],   [0, 0]),         # -> [-2, 8]
    ([[1, 0], [0, 1]], [4, -3],  [11, -13]),      # -> [18, -20]
]


def cmd_smoke(args):
    with open_device() as conn:
        ok = 0
        for i, (w, a, seeds) in enumerate(SMOKE_CASES):
            expected = expected_psums(w, a, seeds)
            result, err = run_one(conn, w, a, seeds)
            label = f"case {i}  W={w} A={a} S={seeds}"
            if err is not None:
                print(f"FAIL  {label}\n      expected {expected}\n      err: {err}")
            elif result == expected:
                ok += 1
                print(f"PASS  {label} -> {result}")
            else:
                print(f"FAIL  {label}\n      expected {expected}\n      got      {result}")

        print(f"---\n{ok}/{len(SMOKE_CASES)} PASS")
        return 0 if ok == len(SMOKE_CASES) else 1


# ----------------------------------------------------------------------------
# subcommand: debug — single case, with RDP sweeps before/after each phase

def hexb(b):
    return " ".join(f"{x:02x}" for x in b)


def rdp_sweep(conn, label):
    """Read all (row, byte) RDPs in one frame; trailing NOP keeps the last
    byte addressable (otherwise the response of the final RDP lands in the
    pipeline slot we wouldn't sample)."""
    pairs = []
    for r in range(ROWS):
        for b in range(PSUM_BYTES):
            pairs.append((pack_ui(CMD_RDP, arg=(r | (b << 1))), 0))
    pairs.append((pack_ui(CMD_NOP), 0))
    resp = x_frame(conn, pairs)
    rdp_bytes = resp[:ROWS * PSUM_BYTES]
    print(f"  [{label}] RDP bytes (r0b0,r0b1,r1b0,r1b1): {hexb(rdp_bytes)}  trailing={resp[-1]:02x}")
    psums = []
    for r in range(ROWS):
        raw = 0
        for b in range(PSUM_BYTES):
            raw |= rdp_bytes[r * PSUM_BYTES + b] << (8 * b)
        psums.append(to_signed16(raw))
    print(f"  [{label}] psums = {psums}")
    return psums


def cmd_debug(args):
    with open_device() as conn:
        # Single case (the previously-failing one — kept as a regression seed).
        w = [[1, 0], [0, 1]]
        a = [4, -3]
        seeds = [11, -13]
        print(f"case: w={w} a={a} seeds={seeds}")
        print(f"  expected post-START: r0=18=0x0012, r1=-20=0xffec")

        reset_chip(conn)

        pre = [(pack_ui(CMD_CLEAR), 0), (pack_ui(CMD_NOP), 0)]
        for r in range(ROWS):
            packed = sum((w[r][c] & 1) << c for c in range(COLS))
            pre.append((pack_ui(CMD_LDW, arg=r), packed))
        for c in range(COLS):
            pre.append((pack_ui(CMD_LDA, arg=c), a[c] & 0xFF))
        pre.append((pack_ui(CMD_NOP), 0))
        x_frame(conn, pre)
        print("after CLEAR/LDW/LDA, before SEED:")
        rdp_sweep(conn, "pre-seed")

        seedframe = []
        for r in range(ROWS):
            sraw = seeds[r] & 0xFFFF
            for b in range(PSUM_BYTES):
                seedframe.append((
                    pack_ui(CMD_SEED, arg=(r | (b << 1))),
                    (sraw >> (8 * b)) & 0xFF,
                ))
        seedframe.append((pack_ui(CMD_NOP), 0))
        x_frame(conn, seedframe)
        print("after SEED, before START:")
        rdp_sweep(conn, "post-seed")

        resp = x_frame(conn, [(pack_ui(CMD_STATUS), 0), (pack_ui(CMD_NOP), 0)])
        print(f"  pre-START status = 0x{resp[0]:02x}  (expect S_START_READY=0x10)")

        runframe = [(pack_ui(CMD_START), 0), (pack_ui(CMD_NOP), 0)]
        polls = 16
        runframe += [(pack_ui(CMD_STATUS), 0)] * polls
        runframe.append((pack_ui(CMD_NOP), 0))
        resp = x_frame(conn, runframe)
        poll_bytes = resp[2:2 + polls]
        print(f"  poll: {hexb(poll_bytes)}  trailing={resp[-1]:02x}")

        print("after START done:")
        rdp_sweep(conn, "post-start")

        return 0


# ----------------------------------------------------------------------------
# subcommand: flash-bitstream — upload a .bin into RP flash + auto-reload

# Mirrors the firmware-side error codes from bitstream_flash.c. 0xff means
# the firmware rejected the request before streaming began (size too large
# or erase failed). 1/2/3 come from bitstream_flash_end.
_FLASH_ERR = {
    0:    "ok",
    1:    "no streaming write in progress (firmware bug)",
    2:    "received fewer bytes than promised",
    3:    "post-write header verification failed",
    0xff: "size invalid or erase failed",
}


def cmd_flash_bitstream(args):
    bits = pathlib.Path(args.bitstream).read_bytes()
    # Generous timeout: erase + per-sector program + the iCE40 SPI reload
    # at the end can together push past 2 s; libusb's default is plenty
    # but we set 15 s for clarity.
    with open_device(timeout_ms=15000) as conn:
        print(f"uploading {len(bits)} bytes from {args.bitstream}")

        # Combine header + payload into one bulk_write so the firmware
        # sees them as one contiguous transfer.
        conn.write(b"F" + struct.pack("<I", len(bits)) + bits)
        rc = conn.read(1)
        if not rc:
            print("FAIL: empty status reply")
            return 1
        code = rc[0]
        msg = _FLASH_ERR.get(code, f"unknown error code 0x{code:02x}")
        if code == 0:
            print(f"flash + reload: {msg}")
            return 0
        print(f"FAIL: rc=0x{code:02x} ({msg})")
        return 1


# ----------------------------------------------------------------------------
# entry point

def main():
    ap = argparse.ArgumentParser(prog="rp_streamer-cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("smoke", help="run hand-crafted matmul cases and verify psums")
    sp.set_defaults(func=cmd_smoke)

    dp = sub.add_parser("debug", help="single-case run with intermediate RDP sweeps")
    dp.set_defaults(func=cmd_debug)

    fp = sub.add_parser("flash-bitstream",
                        help="upload an iCE40 bitstream into RP flash and reload the FPGA")
    fp.add_argument("bitstream", help="path to .bin (e.g. build/<top>.bin)")
    fp.set_defaults(func=cmd_flash_bitstream)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
