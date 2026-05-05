"""
Minimal raw-REPL driver: soft-resets the device, then runs a snippet of
MicroPython code, returns stdout. Avoids mpremote's behavior of resetting
state mid-session.
"""
import sys, time, serial


def run(port: str, code: str, boot_settle_s: float = 3.0, exec_timeout_s: float = 10.0) -> str:
    s = serial.Serial(port, 115200, timeout=2)

    # Interrupt anything running, soft reset.
    s.write(b"\x03\x03\x04")

    # Wait for boot to settle.
    end = time.time() + boot_settle_s
    boot_buf = b""
    while time.time() < end:
        c = s.read(s.in_waiting or 1)
        if c:
            boot_buf += c
            end = time.time() + 1.0

    # Enter raw REPL.
    s.write(b"\x01")
    time.sleep(0.1)
    s.read(s.in_waiting or 1)

    # Send code, then Ctrl-D to execute.
    s.write(code.encode())
    s.write(b"\x04")
    ack = s.read(2)
    if ack != b"OK":
        s.close()
        raise RuntimeError(f"raw repl ACK was {ack!r}, boot tail={boot_buf[-200:]!r}")

    # Read until EOT.
    out = b""
    end = time.time() + exec_timeout_s
    while time.time() < end:
        c = s.read(s.in_waiting or 1)
        if c:
            out += c
            end = time.time() + 0.5
        if b"\x04" in out:
            break

    s.write(b"\x02")  # exit raw REPL
    s.close()
    return out.split(b"\x04")[0].decode("utf-8", "replace")


if __name__ == "__main__":
    code = sys.stdin.read()
    print(run(sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem2101", code))
