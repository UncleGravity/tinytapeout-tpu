"""
Debug helper: drives a single case via 'X' mode, but inserts an extra RDP
sweep BEFORE the START. Splits each phase into its own X-frame so the
response from the very last cmd in each frame is recoverable too.

Used to isolate whether the matmul_streamer bug is in SEED, START/compute,
or RDP path.
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
PSUM_BYTES = 2


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    cs = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not cs:
        sys.exit("no /dev/tty.usbmodem* device")
    return cs[0]


def pack_ui(cmd, arg=0):
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


def x_frame(s, pairs):
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


def reset_assert(s):
    s.write(b"a" + b"\x00" * 4); s.flush()


def reset_release(s):
    s.write(b"d" + b"\x00" * 4); s.flush()


def to_signed16(raw):
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


def hexb(b):
    return " ".join(f"{x:02x}" for x in b)


def rdp_sweep(s, label):
    """Issue CMD_RDP for every (row, byte). Pad with a trailing NOP so the
    last RDP's response actually comes back (otherwise it lands in slot N+1
    which we'd lose if N is the last sample)."""
    pairs = []
    for r in range(ROWS):
        for b in range(PSUM_BYTES):
            pairs.append((pack_ui(CMD_RDP, arg=(r | (b << 1))), 0))
    pairs.append((pack_ui(CMD_NOP), 0))  # trailing flush, in case of pipeline lag
    resp = x_frame(s, pairs)
    rdp_bytes = resp[:ROWS * PSUM_BYTES]
    print(f"  [{label}] RDP raw bytes (per (r,b) order r0b0,r0b1,r1b0,r1b1): {hexb(rdp_bytes)}  trailing={resp[-1]:02x}")
    psums = []
    for r in range(ROWS):
        raw = 0
        for b in range(PSUM_BYTES):
            raw |= rdp_bytes[r * PSUM_BYTES + b] << (8 * b)
        psums.append(to_signed16(raw))
    print(f"  [{label}] decoded psums = {psums}")
    return psums


def main():
    port = find_port()
    s = serial.Serial(port, 115200, timeout=2)
    print(f"open {port}")

    # Case 2 — the failing one.
    w = [[1, 0], [0, 1]]
    a = [4, -3]
    seeds = [11, -13]
    print(f"case: w={w} a={a} seeds={seeds}")
    print(f"  expected seeds in acc_q: r0=0x{11 & 0xFFFF:04x}, r1=0x{-13 & 0xFFFF:04x}")
    print(f"  expected post-START: r0=18=0x0012, r1=-20=0xffec")

    # Reset.
    reset_assert(s)
    x_frame(s, [(pack_ui(CMD_NOP), 0)] * 8)
    reset_release(s)
    x_frame(s, [(pack_ui(CMD_NOP), 0)] * 4)

    # CLEAR and weights/acts.
    pre = []
    pre.append((pack_ui(CMD_CLEAR), 0))
    pre.append((pack_ui(CMD_NOP), 0))
    for r in range(ROWS):
        packed = sum((w[r][c] & 1) << c for c in range(COLS))
        pre.append((pack_ui(CMD_LDW, arg=r), packed))
    for c in range(COLS):
        pre.append((pack_ui(CMD_LDA, arg=c), a[c] & 0xFF))
    pre.append((pack_ui(CMD_NOP), 0))
    x_frame(s, pre)

    print("after CLEAR/LDW/LDA, before SEED:")
    rdp_sweep(s, "pre-seed")

    # SEED.
    seedframe = []
    for r in range(ROWS):
        sraw = seeds[r] & 0xFFFF
        for b in range(PSUM_BYTES):
            seedframe.append((
                pack_ui(CMD_SEED, arg=(r | (b << 1))),
                (sraw >> (8 * b)) & 0xFF,
            ))
    seedframe.append((pack_ui(CMD_NOP), 0))
    x_frame(s, seedframe)

    print("after SEED, before START:")
    rdp_sweep(s, "post-seed")

    # STATUS check.
    resp = x_frame(s, [(pack_ui(CMD_STATUS), 0), (pack_ui(CMD_NOP), 0)])
    print(f"  pre-START status = 0x{resp[0]:02x}  (expect S_START_READY=0x10)")

    # START + poll.
    runframe = [(pack_ui(CMD_START), 0), (pack_ui(CMD_NOP), 0)]
    POLLS = 16
    runframe += [(pack_ui(CMD_STATUS), 0)] * POLLS
    runframe.append((pack_ui(CMD_NOP), 0))  # trailing flush
    resp = x_frame(s, runframe)
    poll_bytes = resp[2:2 + POLLS]
    print(f"  poll responses (one byte per STATUS): {hexb(poll_bytes)}  trailing={resp[-1]:02x}")

    print("after START done:")
    rdp_sweep(s, "post-start")

    s.close()


if __name__ == "__main__":
    main()
