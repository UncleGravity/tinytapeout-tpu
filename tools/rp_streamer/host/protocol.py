"""
Hand-maintained mirror of tools/bonsai_backend/src/protocol.h.

The C++ header is the canonical source. When the C++ side changes (status
bit, opcode, tile shape, encoder layout, X-frame body shape), update this
file in lockstep. The constants and helpers below are organized in the same
order as protocol.h so a side-by-side diff is easy.

Transport: the rp_streamer firmware exposes a USB vendor class with two
raw bulk endpoints. We reach it via pyusb / libusb1. The C++ asic transport
uses libusb the same way.
"""

import struct

import usb.core
import usb.util

# ---------------------------------------------------------------------------
# Tile geometry — mirrors `struct Tile` in protocol.h. Must match RTL params.
TILE_ROWS = 2  # Tile::rows
TILE_COLS = 2  # Tile::cols
TILE_ACT_BITS = 8  # Tile::act_bits
TILE_PSUM_BITS = 16  # Tile::psum_bits

# Aliases used by callers below.
ROWS = TILE_ROWS
COLS = TILE_COLS
PSUM_BYTES = (TILE_PSUM_BITS + 7) // 8

# ---------------------------------------------------------------------------
# Status byte — mirrors `enum Status` in protocol.h.
S_BUSY = 1 << 0  # status_busy
S_DONE = 1 << 1  # status_done
S_WEIGHT_DONE = 1 << 2  # status_weight_done
S_ALL_VALID = 1 << 3  # status_all_valid
S_START_READY = 1 << 4  # status_start_ready
S_IDLE_STABLE = 1 << 5  # status_idle_stable
S_ERROR = 1 << 6  # status_error

# ---------------------------------------------------------------------------
# Command opcodes — mirrors `enum Command` in protocol.h.
CMD_STATUS = 0  # cmd_status
CMD_CLEAR = 1  # cmd_clear
CMD_LDW = 2  # cmd_ldw
CMD_LDA = 3  # cmd_lda
CMD_SEED = 4  # cmd_seed
CMD_START = 5  # cmd_start
CMD_RDP = 6  # cmd_rdp
CMD_NOP = 7  # cmd_nop


# ---------------------------------------------------------------------------
# Encoders — mirror inline functions in protocol.h.
# Argument width math (`row_bits`, `col_bits`) is derived at C++ compile time
# from Tile geometry; we hard-code the result for ROWS=COLS=2 and assert it
# below so accidental drift trips up.

_ROW_BITS = 1  # bit_width_for_count(TILE_ROWS) — mirrors `row_bits` in protocol.h
_COL_BITS = 1  # bit_width_for_count(TILE_COLS) — mirrors `col_bits` in protocol.h

assert TILE_ROWS == 2 and TILE_COLS == 2, (
    "Encoder bit-widths assume ROWS=COLS=2; recompute _ROW_BITS/_COL_BITS if you change Tile."
)


def pack_ui(cmd: int, arg: int = 0) -> int:
    """Mirrors `pack_ui(Command, uint8_t)` in protocol.h."""
    return (cmd & 0x7) | ((arg & 0x1F) << 3)


def to_signed16(raw: int) -> int:
    """Mirrors `sign_extend_psum(uint16_t)` in protocol.h."""
    raw &= 0xFFFF
    return raw - 0x10000 if raw & 0x8000 else raw


# ---------------------------------------------------------------------------
# Transport — vendor-class bulk endpoints over libusb.
#
# Mirrors the C++ asic transport (tools/bonsai_backend/src/transport-usb.cpp):
# same VID/PID defaults, same EP addresses, same X-frame header shape.

USB_VID = 0x2E8A
USB_PID = 0x4002
EP_OUT = 0x01  # host → device (bulk OUT)
EP_IN = 0x81  # device → host (bulk IN)
DEFAULT_TIMEOUT_MS = 5000


class Connection:
    """Bulk transport to the rp_streamer firmware. Use as a context manager."""

    def __init__(self, dev, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.dev = dev
        self.timeout_ms = timeout_ms
        # USB reset clears any stale firmware state from a previous host
        # that disconnected mid-X-frame. Mirrors what UsbTransport does
        # in transport-usb.cpp's open_device().
        try:
            self.dev.reset()
        except usb.core.USBError:
            pass
        self.dev.set_configuration()
        usb.util.claim_interface(self.dev, 0)

    def close(self):
        try:
            usb.util.release_interface(self.dev, 0)
        except usb.core.USBError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, data: bytes) -> None:
        self.dev.write(EP_OUT, data, self.timeout_ms)

    def read(self, n: int) -> bytes:
        return bytes(self.dev.read(EP_IN, n, self.timeout_ms))


def open_device(timeout_ms: int = DEFAULT_TIMEOUT_MS) -> Connection:
    dev = usb.core.find(idVendor=USB_VID, idProduct=USB_PID)
    if dev is None:
        raise IOError(
            f"no rp_streamer device found (VID:PID {USB_VID:04x}:{USB_PID:04x}). "
            f"is the firmware flashed and the cable a data cable?"
        )
    return Connection(dev, timeout_ms=timeout_ms)


# ---------------------------------------------------------------------------
# X-frame helpers. Take a Connection as the first argument.


def x_frame(conn: Connection, pairs) -> bytes:
    """Send one X-frame (5-byte header + 2*N body bytes), read N response bytes."""
    n = len(pairs)
    body = bytearray(2 * n)
    for i, (ui, uio) in enumerate(pairs):
        body[2 * i] = ui & 0xFF
        body[2 * i + 1] = uio & 0xFF
    # Combine header + body into one bulk_write so the firmware sees one
    # contiguous transfer per frame (matches the C++ asic transport).
    conn.write(b"X" + struct.pack("<I", n) + bytes(body))
    return conn.read(n)


def reset_assert(conn: Connection) -> None:
    conn.write(b"a" + b"\x00" * 4)


def reset_release(conn: Connection) -> None:
    conn.write(b"d" + b"\x00" * 4)


def reset_chip(conn: Connection) -> None:
    """Cycle reset: assert, hold for 8 NOPs, release, settle for 4 NOPs."""
    reset_assert(conn)
    x_frame(conn, [(pack_ui(CMD_NOP), 0)] * 8)
    reset_release(conn)
    x_frame(conn, [(pack_ui(CMD_NOP), 0)] * 4)
