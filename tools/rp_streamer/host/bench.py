"""
Modal benchmark: runs T (RP->host), R (host->RP), E (echo) one after another.

Usage:
    python host_bench.py [/dev/tty.usbmodemXXXX]
"""
import glob, struct, sys, threading, time, serial


TOTAL = 4 << 20  # 4 MiB per test
CHUNK = 4096


def find_port():
    if len(sys.argv) > 1:
        return sys.argv[1]
    cs = sorted(glob.glob("/dev/tty.usbmodem*"))
    if not cs:
        sys.exit("no /dev/tty.usbmodem* device")
    return cs[0]


def header(mode, n):
    return mode.encode() + struct.pack("<I", n)


def bench_tx(s, n):
    """RP -> host: RP sends n bytes."""
    s.reset_input_buffer()
    s.write(header("T", n))
    s.flush()
    got = 0
    t0 = time.perf_counter()
    while got < n:
        chunk = s.read(min(CHUNK, n - got))
        if not chunk:
            sys.exit(f"timeout in TX after {got}/{n}")
        got += len(chunk)
    dt = time.perf_counter() - t0
    return got, dt


def bench_rx(s, n):
    """host -> RP: host sends n bytes."""
    s.reset_input_buffer()
    s.write(header("R", n))
    s.flush()
    payload = bytes((i & 0xFF) for i in range(CHUNK))
    sent = 0
    t0 = time.perf_counter()
    while sent < n:
        w = s.write(payload[: n - sent] if n - sent < CHUNK else payload)
        sent += w
    s.flush()
    dt = time.perf_counter() - t0
    return sent, dt


def bench_echo(s, n):
    """Bidir: host sends n, RP echoes n."""
    s.reset_input_buffer()
    s.write(header("E", n))
    s.flush()

    received = bytearray()
    done = threading.Event()

    def reader():
        while len(received) < n and not done.is_set():
            c = s.read(min(CHUNK, n - len(received)) or 1)
            if c:
                received.extend(c)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    payload = bytes((i & 0xFF) for i in range(CHUNK))
    sent = 0
    t0 = time.perf_counter()
    while sent < n:
        w = s.write(payload)
        sent += w
    s.flush()
    t_tx = time.perf_counter() - t0

    t.join(timeout=30)
    done.set()
    t_rt = time.perf_counter() - t0
    return sent, t_tx, len(received), t_rt


def main():
    port = find_port()
    s = serial.Serial(port, 115200, timeout=2)
    print(f"open {port}")

    print("--- TX-only (RP -> host) ---")
    n, dt = bench_tx(s, TOTAL)
    print(f"  {n} B in {dt:.3f} s -> {n/dt/(1<<20):.3f} MiB/s")

    print("--- RX-only (host -> RP) ---")
    n, dt = bench_rx(s, TOTAL)
    print(f"  {n} B in {dt:.3f} s -> {n/dt/(1<<20):.3f} MiB/s")

    print("--- Echo (concurrent both ways) ---")
    sent, t_tx, got, t_rt = bench_echo(s, TOTAL)
    print(f"  host->RP {sent} B in {t_tx:.3f} s -> {sent/t_tx/(1<<20):.3f} MiB/s")
    print(f"  round-trip {got} B in {t_rt:.3f} s -> {got/t_rt/(1<<20):.3f} MiB/s")

    s.close()


if __name__ == "__main__":
    main()
