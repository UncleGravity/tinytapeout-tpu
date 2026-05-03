# rp_streamer

RP2350 firmware + host tooling for streaming weights and activations between the
host (USB) and the TT chip on the ETR demoboard.

Phase 1 (current): pure USB CDC echo for benchmarking the USB-side ceiling.
Phase 2 (next): PIO-driven chip pin protocol mirroring `test/tt_protocol.py`.

## Layout

```
firmware/          C + TinyUSB CDC firmware (pico-sdk, RP2350)
host/bench.py      modal benchmark (T/R/E)
scripts/flash.sh   flash any UF2; auto-BOOTSEL via 'B' or mpremote
firmware_backup/   TT-MicroPython UF2 (needed to load FPGA bitstreams)
shell.nix          dev shell: pico-sdk, arm-gcc, pyserial, mpremote
```

## Findings (USB-only ceiling)

RP2350 native USB is Full-Speed only (12 Mb/s). Numbers below were measured
on an ETR demoboard at `/dev/tty.usbmodem*`, with the C+TinyUSB CDC firmware in
this directory; 4 MiB per test, 4 KB chunks, 4 KB CDC FIFOs.

| Test                            | Throughput     | Notes                       |
|---------------------------------|----------------|-----------------------------|
| TX-only (RP -> host)            | 0.794 MiB/s    | TinyUSB IN-EP scheduling    |
| RX-only (host -> RP)            | 0.976 MiB/s    | ~98% of FS bulk theoretical |
| Echo (concurrent both ways)     | 0.491 each way | aggregate ~0.98 MiB/s       |

For comparison, MicroPython 1.28 (TT-MP) over raw REPL hit only 0.62 MiB/s TX
and 0.15 MiB/s RX (windowed-ack overhead). C+TinyUSB removes that.

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

## Bench

```sh
python host/bench.py                       # auto-detects /dev/tty.usbmodem*
python host/bench.py /dev/tty.usbmodem...  # explicit
```

Runs three tests in sequence: TX-only, RX-only, then concurrent echo.
Each test sends a 5-byte header `[mode][n_le_u32]` and proceeds.

## Modal protocol (firmware/main.c)

```
Header: 5 bytes
  mode : u8        'T' tx N bytes from RP, 'R' drop N from host,
                   'E' echo N bytes both ways, 'B' reboot to BOOTSEL.
  n    : u32 LE    payload length for this test (ignored for 'B').
```

The firmware loops in mode-select forever, so multiple tests can run on a
single CDC connection without rebooting between them.

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
7. `python host/bench.py` — bench (current) or stream (phase 2).

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
