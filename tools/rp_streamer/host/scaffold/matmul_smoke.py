"""
FPGA bring-up scaffolding (NOT a regression test).

Drives tt_um_unclegravity_tpu via the TT-MicroPython REPL on the FPGA
breakout. Useful before flashing the rp_streamer C firmware to confirm the
bitstream landed and the chip's ui_in/uio/uo_out path is intact. Production
testing uses bonsai-matmul-smoke (verilator) and host/cli.py (firmware).

Sends a tiny test vector (LDW + LDA + SEED + START + RDP), verifies the
output partial sums match the dot_ref model from test/common.py:

    psum[r] = seed[r] + sum( act[c] if w[r][c] else -act[c] for c in range(cols) )

Bypasses the SDK's contention check on ui_in pins (FPGA pads pull HIGH which
otherwise blocks the RP from driving them).

Usage:
    python host/scaffold/matmul_smoke.py [/dev/tty.usbmodemXXXX]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from repl_helper import run


PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem2101"


# Two test cases. (weights[rows][cols], acts[cols], seeds[rows]).
# expected[r] = seed[r] + sum( act[c] if w[r][c] else -act[c] )
CASES = [
    # all-positive weights
    ([[1, 1], [1, 1]], [10, 20], [0, 0]),       # -> [30, 30]
    # mixed
    ([[1, 0], [1, 1]], [3, 5],   [0, 0]),       # -> [-2, 8]
    # with seeds
    ([[1, 0], [0, 1]], [4, -3],  [11, -13]),    # -> 11+4-(-3) ; -13-4+(-3) = [18, -20]
]


# Inline MicroPython program. It receives the test cases via a leading
# Python literal we string-substitute in.
DEVICE_CODE = r"""
from ttboard.demoboard import DemoBoard
from ttboard.mode import RPMode
import ttboard.util.platform as platform

CASES = {cases}

CMD_STATUS = 0
CMD_CLEAR  = 1
CMD_LDW    = 2
CMD_LDA    = 3
CMD_SEED   = 4
CMD_START  = 5
CMD_RDP    = 6
CMD_NOP    = 7

S_DONE        = 1 << 1
S_ALL_VALID   = 1 << 3
S_START_READY = 1 << 4
S_ERROR       = 1 << 6

ROWS = 2
COLS = 2
PSUM_BYTES = 2  # PSUM_WIDTH=16

tt = DemoBoard.get()

# Disable the contention guard so we can drive ui_in even though the FPGA
# pulls them HIGH when undriven.
tt.pins.dieOnInputControlSwitchHigh = False
tt.mode = RPMode.ASIC_RP_CONTROL
tt.clock_project_stop()

# In ASIC_RP_CONTROL the bidir uio pins default to "input from RP" -- we
# carry the data byte over uio_in[7:0], so we have to flip RP's bidir OE
# to drive all 8 bits.
tt.uio_oe_pico.value = 0xFF

# Probe: confirm the RP can actually drive these pins (read back what we
# wrote via the RP's own input register).
import ttboard.util.platform as platform
platform.write_ui_in_byte(0xA5)
platform.write_uio_byte(0x5A)
print('writeback ui_in=0x%02x uio=0x%02x (expect 0xa5/0x5a)' % (
    platform.read_ui_in_byte(),
    platform.read_uio_byte(),
))

def send(c, data=0, arg=0):
    ui = (c & 0x7) | ((arg & 0x1F) << 3)
    platform.write_ui_in_byte(ui)
    platform.write_uio_byte(data & 0xFF)
    tt.clock_project_once()

def status():
    send(CMD_STATUS)
    return platform.read_uo_out_byte()

def to_signed16(raw):
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw

def to_u8(v):
    return v & 0xFF

def reset_chip():
    tt.reset_project(True)
    for _ in range(8):
        tt.clock_project_once()
    tt.reset_project(False)
    for _ in range(2):
        tt.clock_project_once()

def run_one(weights, acts, seeds):
    reset_chip()
    send(CMD_CLEAR); send(CMD_NOP)

    # weights: pack COLS bits per row
    for r in range(ROWS):
        packed = 0
        for c in range(COLS):
            packed |= (weights[r][c] & 1) << c
        send(CMD_LDW, data=packed, arg=r)

    # activations: one int8 per col
    for c in range(COLS):
        send(CMD_LDA, data=to_u8(acts[c]), arg=c)

    # seeds: PSUM_BYTES per row
    for r in range(ROWS):
        sraw = seeds[r] & 0xFFFF
        for b in range(PSUM_BYTES):
            send(CMD_SEED, data=(sraw >> (8 * b)) & 0xFF, arg=r | (b << 1))

    pre = status()
    if not (pre & S_START_READY):
        return ('FAIL precond', 'pre-START status=0x%02x missing START_READY' % pre)
    if pre & S_ERROR:
        return ('FAIL precond', 'pre-START status=0x%02x ERROR set' % pre)

    send(CMD_START); send(CMD_NOP)

    st = 0
    for _ in range(64):
        st = status()
        if st & S_ERROR:
            return ('FAIL run', 'status=0x%02x ERROR' % st)
        if st & S_DONE:
            break
    else:
        return ('FAIL run', 'no DONE within 64 polls; last=0x%02x' % st)

    if not (st & S_ALL_VALID):
        return ('FAIL run', 'DONE without ALL_VALID; status=0x%02x' % st)

    # read psums
    out = []
    for r in range(ROWS):
        raw = 0
        for b in range(PSUM_BYTES):
            send(CMD_RDP, arg=r | (b << 1))
            raw |= platform.read_uo_out_byte() << (8 * b)
        out.append(to_signed16(raw))
    return ('OK', out)

ok = 0
for i, (w, a, s) in enumerate(CASES):
    expected = []
    for r in range(ROWS):
        v = s[r]
        for c in range(COLS):
            v += a[c] if w[r][c] else -a[c]
        expected.append(v)
    label = 'case %d  W=%s A=%s S=%s' % (i, w, a, s)
    state, payload = run_one(w, a, s)
    if state == 'OK' and payload == expected:
        ok += 1
        print('PASS  %s -> %s' % (label, payload))
    else:
        print('FAIL  %s' % label)
        print('      expected %s' % expected)
        print('      got      state=%r payload=%r' % (state, payload))

print('---')
print('%d/%d PASS' % (ok, len(CASES)))
"""


def main():
    code = DEVICE_CODE.format(cases=repr(CASES))
    out = run(PORT, code, boot_settle_s=4.0, exec_timeout_s=20.0)
    print(out)


if __name__ == "__main__":
    main()
