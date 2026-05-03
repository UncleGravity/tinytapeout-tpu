"""
Matmul smoke test via the rp_streamer C firmware.

Drives the chip directly through the host->RP USB CDC link, using mode 'X'
(streamed transactions). Each X-cycle is one (ui_in, uio_in) write + one
clock pulse + one uo_out sample.

Prerequisite: the FPGA must already be configured with tt_um_unclegravity_tpu
(load via TT-MicroPython, then BOOTSEL to rp_streamer; iCE40 SRAM persists
across the swap as long as USB power stays).

Usage:
    python host/matmul_streamer.py [/dev/tty.usbmodemXXXX]
"""
import glob
import struct
import sys
import serial


CMD_STATUS, CMD_CLEAR, CMD_LDW, CMD_LDA, CMD_SEED, CMD_START, CMD_RDP, CMD_NOP = range(8)

S_DONE        = 1 << 1
S_ALL_VALID   = 1 << 3
S_START_READY = 1 << 4
S_ERROR       = 1 << 6

ROWS = 2
COLS = 2
PSUM_BYTES = 2     # PSUM_WIDTH = 16 bits


def find_port() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    cs = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not cs:
        sys.exit("no /dev/tty.usbmodem* device")
    return cs[0]


def pack_ui(cmd: int, arg: int = 0) -> int:
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


def x_frame(s: serial.Serial, pairs) -> bytes:
    n = len(pairs)
    s.write(b"X" + struct.pack("<I", n))
    body = bytearray(2 * n)
    for i, (ui, uio) in enumerate(pairs):
        body[2 * i] = ui & 0xFF
        body[2 * i + 1] = uio & 0xFF
    s.write(bytes(body))
    s.flush()
    out = b""
    while len(out) < n:
        chunk = s.read(n - len(out))
        if not chunk:
            raise IOError(f"X read timeout: got {len(out)}/{n}")
        out += chunk
    return out


def reset_assert(s: serial.Serial) -> None:
    s.write(b"a" + b"\x00" * 4)
    s.flush()


def reset_release(s: serial.Serial) -> None:
    s.write(b"d" + b"\x00" * 4)
    s.flush()


def to_signed16(raw: int) -> int:
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def run_one(s: serial.Serial, w, a, seeds):
    # Cycle reset.
    reset_assert(s)
    x_frame(s, [(pack_ui(CMD_NOP), 0)] * 8)
    reset_release(s)
    x_frame(s, [(pack_ui(CMD_NOP), 0)] * 4)

    # Setup: CLEAR, LDW, LDA, SEED, then a STATUS check.
    setup = []
    setup.append((pack_ui(CMD_CLEAR), 0))
    setup.append((pack_ui(CMD_NOP), 0))
    for r in range(ROWS):
        packed = sum((w[r][c] & 1) << c for c in range(COLS))
        setup.append((pack_ui(CMD_LDW, arg=r), packed))
    for c in range(COLS):
        setup.append((pack_ui(CMD_LDA, arg=c), a[c] & 0xFF))
    for r in range(ROWS):
        sraw = seeds[r] & 0xFFFF
        for b in range(PSUM_BYTES):
            setup.append((
                pack_ui(CMD_SEED, arg=(r | (b << 1))),
                (sraw >> (8 * b)) & 0xFF,
            ))
    setup.append((pack_ui(CMD_STATUS), 0))

    setup_resp = x_frame(s, setup)
    pre_status = setup_resp[-1]
    if pre_status & S_ERROR:
        return None, f"pre-START ERROR set, status=0x{pre_status:02x}"
    if not (pre_status & S_START_READY):
        return None, f"pre-START missing START_READY, status=0x{pre_status:02x}"

    # START + poll + RDP, all in one frame.
    POLLS = 16
    runframe = []
    runframe.append((pack_ui(CMD_START), 0))
    runframe.append((pack_ui(CMD_NOP), 0))
    for _ in range(POLLS):
        runframe.append((pack_ui(CMD_STATUS), 0))
    for r in range(ROWS):
        for b in range(PSUM_BYTES):
            runframe.append((pack_ui(CMD_RDP, arg=(r | (b << 1))), 0))

    resp = x_frame(s, runframe)
    poll_resp = resp[2:2 + POLLS]
    rdp_resp = resp[2 + POLLS:]

    if not any(st & S_DONE for st in poll_resp):
        return None, f"no DONE in {POLLS} polls; last=0x{poll_resp[-1]:02x}"
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


def main():
    port = find_port()
    s = serial.Serial(port, 115200, timeout=2)
    print(f"open {port}")

    cases = [
        ([[1, 1], [1, 1]], [10, 20], [0, 0]),         # -> [30, 30]
        ([[1, 0], [1, 1]], [3, 5],   [0, 0]),         # -> [-2, 8]
        ([[1, 0], [0, 1]], [4, -3],  [11, -13]),      # -> [18, -20]
    ]

    ok = 0
    for i, (w, a, seeds) in enumerate(cases):
        expected = []
        for r in range(ROWS):
            v = seeds[r]
            for c in range(COLS):
                v += a[c] if w[r][c] else -a[c]
            expected.append(v)

        result, err = run_one(s, w, a, seeds)
        label = f"case {i}  W={w} A={a} S={seeds}"
        if err is not None:
            print(f"FAIL  {label}\n      expected {expected}\n      err: {err}")
        elif result == expected:
            ok += 1
            print(f"PASS  {label} -> {result}")
        else:
            print(f"FAIL  {label}\n      expected {expected}\n      got      {result}")

    print(f"---\n{ok}/{len(cases)} PASS")
    s.close()


if __name__ == "__main__":
    main()
