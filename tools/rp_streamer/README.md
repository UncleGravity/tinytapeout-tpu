# rp_streamer

RP2350 firmware + host tooling for driving the TT chip from the host over USB
on the ETR demoboard. The firmware exposes the chip's per-cycle (ui_in,
uio_in) → uo_out interface as a streamed transaction protocol; the bonsai
backend (tools/bonsai_backend/) is the production consumer.

## Layout

```
firmware/                C + TinyUSB vendor firmware (pico-sdk, RP2350)
host/protocol.py         shared chip-cycle constants + X-frame helpers
host/cli.py              host CLI: subcommands `smoke`, `debug`
host/scaffold/           FPGA bring-up via TT-MicroPython REPL (one-shot)
scripts/flash.sh         flash any UF2; auto-BOOTSEL via 'B' or mpremote
firmware_backup/         TT-MicroPython UF2 (needed to load FPGA bitstreams)
shell.nix                dev shell: pico-sdk, arm-gcc, pyserial, mpremote
```

## USB ceiling (historical — measured during firmware bring-up)

RP2350 native USB is Full-Speed only (12 Mb/s). Earlier diagnostic firmware
modes (T tx, R rx, E echo) measured the wire ceiling at ~0.98 MiB/s in each
direction (4 KB chunks, 4 KB FIFOs). Those modes have been removed; the table
below is kept as design context.

| Test                            | Throughput     | Notes                       |
|---------------------------------|----------------|-----------------------------|
| TX-only (RP -> host)            | 0.794 MiB/s    | TinyUSB IN-EP scheduling    |
| RX-only (host -> RP)            | 0.976 MiB/s    | ~98% of FS bulk theoretical |
| Echo (concurrent both ways)     | 0.491 each way | aggregate ~0.98 MiB/s       |

**USB is the wall.** RP2350 has no built-in HS PHY; the ETR doesn't route an
external ULPI. The chip itself runs at 2-50 MHz, well above what USB can feed
byte-for-byte, so any reasonable batched protocol over PIO will keep the chip
fed without USB becoming idle.

## Dev shell

```sh
cd tools/rp_streamer
nix-shell                 # pulls pico-sdk (with submodules), arm-gcc, python+pyserial, mpremote
```

## Build

```sh
cd firmware
mkdir -p build && cd build
cmake -G Ninja ..
ninja
# -> build/rp_echo.uf2
```

## Flash

The C firmware accepts a `'B'` byte (with a 4-byte payload) over CDC to reboot
into BOOTSEL programmatically; TT-MicroPython is reachable via `mpremote
machine.bootloader()`. `scripts/flash.sh` tries both paths.

```sh
./scripts/flash.sh                 # flash firmware/build/rp_echo.uf2
./scripts/flash.sh ttmp            # restore firmware_backup/tt-demo-rp2350-v3.0.6.uf2
./scripts/flash.sh /path/x.uf2     # custom
```

If automatic BOOTSEL fails (e.g., no firmware is responding on either path):
hold the BOOT button on the demoboard, tap RESET, release BOOT, then re-run.

`cat`, not `cp`: macOS's fskit FAT16 mount of the BOOTSEL drive blocks `cp`'s
metadata copy under sandboxing, but `cat <uf2> > /Volumes/RP2350/...` works.

## Wire protocol (firmware/main.c)

```
Header: 5 bytes
  mode : u8        'X' streamed transactions (body follows),
                   'a' assert chip reset, 'd' deassert chip reset,
                   'B' reboot RP into USB BOOTSEL,
                   'F' upload N bytes of iCE40 bitstream into RP flash,
                   'L' re-load the stored bitstream into the FPGA.
  n    : u32 LE    payload length for 'X' / 'F'; ignored for a/d/B/L.

'X' body:  2*n bytes [ui_in, uio_in]+; reply: n bytes (uo_out).
'F' body:  n bytes of bitstream;       reply: 1 byte (0 = ok, nonzero = err).
'L' body:  none;                       reply: 1 byte (0 = ok, 1 = no
           bitstream stored, 2 = load failed).
```

The firmware loops in mode-select forever; the host (bonsai backend or
`host/` scripts) drives one or more frames on a single connection.

### Bitstream auto-load on power-up

On boot the firmware checks for a valid bitstream in the top 256 KB of
flash (header magic `BTSB`); if present, it streams it into the iCE40 over
slave-SPI before entering the mode loop. **Surviving a power cycle no
longer needs TT-MicroPython** — flash the bitstream once via `'F'`, and
every subsequent power-on auto-restores it.

Pin map (FabricFox v2 SPI side, copied verbatim from
`tt-micropython-firmware/src/ttboard/fpga/fabricfoxv2.py`):

```
CRESET_B = GPIO 1
MOSI     = GPIO 3
SS       = GPIO 5
SCK      = GPIO 6   (hardware spi0)
```

Workflow:

```sh
nix run .#fpga                                                    # build/<top>.bin
python tools/rp_streamer/host/cli.py flash-bitstream build/<top>.bin
```

The firmware writes to flash, re-loads the iCE40 immediately, and the
chip is alive — no TT-MP, no soft-reboot dance.

## Dev loop with the FabricFox FPGA breakout

The iCE40 on FabricFox is SRAM-based, with no on-board flash. TT-MicroPython
detects the FPGA breakout (`DemoboardCarrier.FPGA`), then on each soft-reboot
streams a bitstream from `:/bitstreams/<name>.bin` into the iCE40 over PIO+SPI
(`ttboard.fpga.fabricfoxv2.spi_transferPIO`). To get the latest RTL onto the
FPGA:

1. `./scripts/flash.sh ttmp` — TT-MP running on the RP2350.
2. Build the bitstream (if not already built): `nix run .#fpga`
   (runs `tt_fpga.py harden`, output: `<repo>/build/<top>.bin`).
3. Upload + set as default + clockrate, all in one:
   ```
   ./tt/tt_fpga.py configure --port /dev/tty.usbmodem* \
       --upload --set-default --clockrate 12000000
   ```
   - `--upload` copies `build/<top>.bin` into `:/bitstreams/<top>.bin`.
   - `--set-default` writes the project name into `[DEFAULT].project` in `config.ini`.
   - `--clockrate` writes the clockrate into `[<top>].clock_frequency` in `config.ini`.
   - On the next soft-reboot the SDK auto-loads this bitstream into the FPGA.
4. **Soft-reboot the RP2350** (Ctrl-C+Ctrl-D over serial; mpremote's `reset`
   doesn't always retrigger the boot probe path that picks up the FPGA).
   Watch the boot log for `Transmission complete, total bytes: ...` from
   `fabricfoxv2.spi_transferPIO` — that's the bitstream landing in the iCE40.
5. **Don't unplug USB.** iCE40 SRAM keeps the bitstream across a subsequent
   RP2350 firmware swap to `rp_streamer`, as long as USB power stays.
6. `./scripts/flash.sh` — rp_streamer takes over; bitstream is intact.
7. `python host/cli.py smoke` — run hand-crafted matmul cases against the chip.

To enable the project at all (no `--set-default`), in the REPL after step 4:
```
tt.shuttle.<name>.enable()    # streams the bitstream and selects it
```

Notes:
- TT-MP's default mode for FPGA carriers is `ASIC_MANUAL_INPUTS` (DIPs drive
  the inputs). To drive `ui_in`/`uio` from the RP2 instead, set
  `tt.mode = RPMode.ASIC_RP_CONTROL` (this stops auto-clocking — re-enable
  with `tt.clock_project_PWM(<hz>)` or step manually with `clock_project_once()`).
- Updating the RTL means re-running step 2 (rebuild) and step 3+4 (upload + reboot).
- Eventually, porting `fabricfoxv2.spi_transferPIO` to the C firmware would
  remove the bounce-to-TT-MP step, but it's not on the "saturate USB"
  critical path.

### Smoke test (chip is alive)

After steps 1–4, with `tt_um_unclegravity_tpu` loaded:
```
tt.mode = RPMode.ASIC_RP_CONTROL
import ttboard.util.platform as platform
platform.write_ui_in_byte(0)            # cmd=STATUS, arg=0
platform.read_uo_out_byte()             # -> 0x30 (status byte)
```
A non-`0xFF` reading on `uo_out` confirms the FPGA is running the design and
the RP↔FPGA pin path is intact.
