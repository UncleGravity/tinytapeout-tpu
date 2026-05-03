#!/usr/bin/env bash
# Flash a UF2 to the RP2350 demoboard.
#
#   ./scripts/flash.sh               -> firmware/build/rp_echo.uf2  (default)
#   ./scripts/flash.sh streamer      -> same as above
#   ./scripts/flash.sh ttmp          -> firmware_backup/tt-demo-rp2350-v3.0.6.uf2
#   ./scripts/flash.sh /path/x.uf2   -> custom
#
# Tries to enter BOOTSEL automatically:
#   1. Send 'B' header to a running rp_streamer firmware on CDC, OR
#   2. mpremote machine.bootloader() to a running TT-MicroPython.
# Falls back to manual BOOTSEL (hold BOOT, tap RESET).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-streamer}" in
    streamer) UF2="$ROOT/firmware/build/rp_echo.uf2" ;;
    ttmp)     UF2="$ROOT/firmware_backup/tt-demo-rp2350-v3.0.6.uf2" ;;
    *)        UF2="$1" ;;
esac

[ -f "$UF2" ] || { echo "missing UF2: $UF2"; exit 1; }
echo "flashing: $UF2"

PORT=$(ls /dev/tty.usbmodem* 2>/dev/null | head -1 || true)

if [ -n "$PORT" ]; then
    # Vendor-class rp_streamer no longer exposes /dev/tty.usbmodem*, so this
    # path only fires when TT-MicroPython is currently running. The
    # vendor-class bootsel reset is intentionally not exposed via libusb here:
    # if you're already running the bonsai backend you can call its driver,
    # and otherwise the manual BOOT+RESET fallback below is one tap.
    nix-shell -p mpremote --run "mpremote connect $PORT exec 'import machine; machine.bootloader()'" >/dev/null 2>&1 || true
fi

echo "waiting for RP2350 BOOTSEL drive..."
for i in $(seq 1 30); do
    for v in "/Volumes/RP2350" "/Volumes/RPI-RP2" "/Volumes/RP2"; do
        if [ -d "$v" ]; then
            echo "found $v"
            # cp fails under macOS sandbox on the fskit FAT16 mount; cat works.
            cat "$UF2" > "$v/$(basename "$UF2")"
            sync
            echo "copied. board will reboot."
            exit 0
        fi
    done
    sleep 1
done

echo "no BOOTSEL drive after 30s." >&2
echo "manual fallback: hold BOOT, tap RESET, release BOOT, then re-run." >&2
exit 1
