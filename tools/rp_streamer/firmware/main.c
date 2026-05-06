#include <stdint.h>
#include "pico/stdlib.h"
#include "pico/bootrom.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "hardware/structs/sio.h"
#include "tusb.h"

#include "chip_cycle.pio.h"
#include "ice40_loader.h"
#include "bitstream_flash.h"

// Protocol: host sends 5-byte header [mode][n_le_u32] per request.
//
//   'B' -> RP reboots into USB BOOTSEL (n ignored)
//   'a' -> assert chip reset (rst_n LOW), n ignored
//   'd' -> deassert chip reset (rst_n HIGH), n ignored
//   'X' -> stream N transactions. Body: 2*N bytes [ui_in, uio_in]+. Reply: N
//          bytes (uo_out sampled after each clock edge).
//   'M' -> matmul macro mode. Body: 5*N tile descriptors (see below). The
//          firmware unrolls each descriptor into the chip-cycle stream
//          locally, so the wire only carries the tile inputs (~5 bytes)
//          instead of the full per-cycle stimulus (~36 bytes for an
//          intermediate tile). Reply: 4 bytes per descriptor whose
//          ends_run flag is set (psum row 0 LE u16, psum row 1 LE u16);
//          mid-run tiles produce no bytes on the wire.
//   'F' -> stream N bytes of iCE40 bitstream into RP2350 flash; on
//          completion, also re-loads it into the FPGA. Reply: 1 byte
//          status (0=ok, nonzero=error code from bitstream_flash_end).
//   'L' -> re-load the bitstream currently stored in flash into the FPGA.
//          Reply: 1 byte (0=ok, 1=no bitstream stored, 2=load failed).
//
// 'M' tile descriptor (5 bytes):
//   byte[0] = flags (bit 0 = starts_run, bit 1 = ends_run)
//   byte[1] = packed_weights[row 0]   (1 bit per col)
//   byte[2] = packed_weights[row 1]
//   byte[3] = acts[col 0] (int8 cast to u8)
//   byte[4] = acts[col 1]
// Seeds are always zero (CLEAR at the run head zeros acc_q anyway). If
// non-zero seeds become needed, extend the protocol with a flag bit + 4
// optional seed bytes per starts_run tile.
//
// Pin map (ETR demoboard, chip side):
//   rst_n      = GPIO14                 (bank 0, SIO)
//   clk        = GPIO16                 (bank 0, PIO sideset)
//   ui_in[0:7] = GPIO17:24              (bank 0, PIO OUT)
//   uio[0:7]   = GPIO25:32              (uio[7]=GPIO32 is bank 1, PIO OUT)
//   uo_out[0:7]= GPIO33:40              (bank 1, PIO IN)
//
// Pin map (FabricFox v2, iCE40 slave-SPI side — see ice40_loader.h):
//   CRESET_B   = GPIO 1
//   MOSI       = GPIO 3
//   SS         = GPIO 5
//   SCK        = GPIO 6
//
// One PIO SM with gpio_base=16 covers GPIOs 16-47 in a single 32-pin window,
// so OUT/IN can drive across the bank-0/bank-1 boundary atomically.

#define PIN_RESET    14
#define PIN_CLOCK    16
#define PIN_UI_BASE  17
#define PIN_UIO_BASE 25
#define PIN_UO_BASE  33

static PIO  chip_pio;
static uint chip_sm;

static void chip_pins_init(void) {
    // rst_n stays on SIO — GPIO14 is below the PIO 16-47 window.
    gpio_init(PIN_RESET);
    gpio_set_dir(PIN_RESET, GPIO_OUT);
    gpio_put(PIN_RESET, 1);

    chip_pio = pio0;
    pio_set_gpio_base(chip_pio, 16);
    chip_sm = (uint) pio_claim_unused_sm(chip_pio, true);
    uint offset = pio_add_program(chip_pio, &chip_cycle_program);
    chip_cycle_program_init(chip_pio, chip_sm, offset,
                            PIN_CLOCK, PIN_UI_BASE, PIN_UO_BASE);
}

static inline uint8_t chip_cycle(uint8_t ui, uint8_t uio) {
    pio_sm_put_blocking(chip_pio, chip_sm,
                        ((uint32_t) uio << 8) | (uint32_t) ui);
    return (uint8_t) pio_sm_get_blocking(chip_pio, chip_sm);
}

// ---------------------------------------------------------------------------
// 'M' mode: tile macro expansion (mirrors src/protocol.h on the host).
//
// Tile geometry. ROWS = 2, COLS = 2 are baked into the chip's RTL today so
// they're hardcoded here too — bumping the array size means a new bitstream
// AND a new firmware build, no point parameterizing one without the other.
// ---------------------------------------------------------------------------

#define TILE_ROWS       2
#define TILE_COLS       2
#define TILE_PSUM_BYTES 2  /* 16-bit psum, two bytes per row */
#define TILE_DONE_PAD   6  /* matches src/protocol.h tile_done_pad_cycles */

#define ROW_BITS        1  /* $clog2(TILE_ROWS) */

// Command opcodes. Must match src/rtl/tpu_cmd_decode.v and src/protocol.h.
enum {
    CMD_STATUS = 0,
    CMD_CLEAR  = 1,
    CMD_LDW    = 2,
    CMD_LDA    = 3,
    CMD_SEED   = 4,
    CMD_START  = 5,
    CMD_RDP    = 6,
    CMD_NOP    = 7,
};

#define TILE_FLAG_STARTS_RUN 0x01u
#define TILE_FLAG_ENDS_RUN   0x02u

static inline uint8_t pack_ui(uint8_t cmd, uint8_t arg) {
    return (uint8_t) ((cmd & 0x7u) | ((arg & 0x1fu) << 3));
}

// Drive one tile's chip-cycle sequence. Mirrors host-side build_tile_pairs;
// the chip can't tell whether the bytes were unrolled here or sent over USB
// per cycle, so the verilator parity test still covers this path.
static void run_tile_macro(uint8_t flags,
                           uint8_t pw0, uint8_t pw1,
                           uint8_t a0,  uint8_t a1,
                           uint8_t * psum_out) {
    const uint8_t starts_run = flags & TILE_FLAG_STARTS_RUN;
    const uint8_t ends_run   = flags & TILE_FLAG_ENDS_RUN;

    if (starts_run) {
        chip_cycle(pack_ui(CMD_CLEAR, 0), 0);
        chip_cycle(pack_ui(CMD_NOP,   0), 0);
    }
    chip_cycle(pack_ui(CMD_LDW, 0), pw0);
    chip_cycle(pack_ui(CMD_LDW, 1), pw1);
    chip_cycle(pack_ui(CMD_LDA, 0), a0);
    chip_cycle(pack_ui(CMD_LDA, 1), a1);
    if (starts_run) {
        for (int r = 0; r < TILE_ROWS; r++) {
            for (int b = 0; b < TILE_PSUM_BYTES; b++) {
                const uint8_t arg = (uint8_t) ((r & 1) | ((b & 1) << ROW_BITS));
                chip_cycle(pack_ui(CMD_SEED, arg), 0);
            }
        }
    }
    chip_cycle(pack_ui(CMD_START, 0), 0);
    chip_cycle(pack_ui(CMD_NOP,   0), 0);
    for (int i = 0; i < TILE_DONE_PAD; i++) {
        chip_cycle(pack_ui(CMD_NOP, 0), 0);
    }
    if (ends_run && psum_out != NULL) {
        for (int r = 0; r < TILE_ROWS; r++) {
            for (int b = 0; b < TILE_PSUM_BYTES; b++) {
                const uint8_t arg = (uint8_t) ((r & 1) | ((b & 1) << ROW_BITS));
                psum_out[r * TILE_PSUM_BYTES + b] =
                    chip_cycle(pack_ui(CMD_RDP, arg), 0);
            }
        }
    }
}

static void usb_read_blocking(uint8_t *p, uint32_t n) {
    uint32_t got = 0;
    while (got < n) {
        tud_task();
        if (tud_vendor_available()) {
            got += tud_vendor_read(p + got, n - got);
        }
    }
}

static void usb_write_blocking(const uint8_t *p, uint32_t n) {
    uint32_t put = 0;
    while (put < n) {
        tud_task();
        uint32_t w = tud_vendor_write(p + put, n - put);
        if (w) put += w;
        else tud_vendor_write_flush();
    }
}

int main(void) {
    // Overclock from the SDK's default 150 MHz to 180 MHz so the PIO can
    // drive the chip at ~45 MHz (4-instruction loop × 5.56 ns ≈ 22 ns
    // chip cycle). The FPGA bitstream is hardened at TT_FPGA_FREQ=45 with
    // ~6 MHz timing margin (nextpnr Fmax ~51 MHz). RP2350's spec range
    // covers this sysclk comfortably; if the chip flakes, drop to 150 MHz
    // and `[1]` delays in chip_cycle.pio for ~30 MHz chip cycle.
    set_sys_clock_khz(180000, true);

    tud_init(0);
    chip_pins_init();
    ice40_pins_init();

    // Auto-load the stored bitstream (if any) into the iCE40 before the
    // host comes up. Lets the chip survive a power cycle without TT-MP.
    if (bitstream_flash_present()) {
        ice40_load_bitstream(bitstream_flash_data(), bitstream_flash_length());
    }

    while (1) {
        uint8_t hdr[5];
        usb_read_blocking(hdr, sizeof(hdr));
        uint8_t mode = hdr[0];
        uint32_t n = (uint32_t) hdr[1]
                   | ((uint32_t) hdr[2] << 8)
                   | ((uint32_t) hdr[3] << 16)
                   | ((uint32_t) hdr[4] << 24);

        if (mode == 'B') {
            tud_vendor_write_flush();
            sleep_ms(50);
            reset_usb_boot(0, 0);
        } else if (mode == 'a') {
            sio_hw->gpio_clr = 1u << PIN_RESET;
        } else if (mode == 'd') {
            sio_hw->gpio_set = 1u << PIN_RESET;
        } else if (mode == 'F') {
            // Streaming flash write. Validate up front; reply once at end.
            if (n == 0 || n > BITSTREAM_MAX_BYTES || !bitstream_flash_begin(n)) {
                const uint8_t resp = 0xff;  // size invalid / erase failed
                usb_write_blocking(&resp, 1);
                tud_vendor_write_flush();
                continue;
            }
            uint32_t remaining = n;
            uint8_t  rx[1024];
            while (remaining > 0) {
                uint32_t want = remaining < sizeof(rx) ? remaining : sizeof(rx);
                usb_read_blocking(rx, want);
                bitstream_flash_append(rx, want);
                remaining -= want;
            }
            const int rc = bitstream_flash_end();
            if (rc == 0) {
                // Re-load the freshly-written bitstream into the iCE40 so
                // the chip is alive without a power cycle.
                ice40_load_bitstream(bitstream_flash_data(),
                                     bitstream_flash_length());
            }
            const uint8_t resp = (uint8_t) rc;
            usb_write_blocking(&resp, 1);
            tud_vendor_write_flush();
        } else if (mode == 'L') {
            uint8_t resp;
            if (!bitstream_flash_present()) {
                resp = 1;
            } else {
                resp = ice40_load_bitstream(bitstream_flash_data(),
                                            bitstream_flash_length()) ? 0 : 2;
            }
            usb_write_blocking(&resp, 1);
            tud_vendor_write_flush();
        } else if (mode == 'X') {
            // Process up to CHUNK cycles at a time: read 2*CHUNK pair bytes,
            // run CHUNK chip cycles into a local response buffer, write
            // CHUNK bytes once. Per-byte tud_vendor_write/tud_task overhead
            // dominated the previous 1-byte-at-a-time loop and was the
            // floor on USB-FS effective throughput (~133 µs/tile asymptote
            // → expected ~50 µs at the wire-rate ceiling).
            enum { CHUNK = 64 };
            uint8_t in_buf[2 * CHUNK];
            uint8_t out_buf[CHUNK];
            uint32_t remaining = n;
            while (remaining > 0) {
                const uint32_t batch = remaining < CHUNK ? remaining : CHUNK;
                usb_read_blocking(in_buf, 2 * batch);
                for (uint32_t i = 0; i < batch; i++) {
                    out_buf[i] = chip_cycle(in_buf[2 * i], in_buf[2 * i + 1]);
                }
                usb_write_blocking(out_buf, batch);
                remaining -= batch;
            }
            tud_vendor_write_flush();
        } else if (mode == 'M') {
            // Tile macro mode. Each iteration reads up to CHUNK 5-byte tile
            // descriptors, unrolls them into the chip-cycle stream, and
            // sends back 4 bytes per ends_run tile. CHUNK is sized so the
            // tightest inner loop (descriptors → chip_cycle calls) stays
            // under ~1 ms wall time, which is well within tud_task()'s
            // tolerance for not being called.
            enum { CHUNK = 32 };
            uint8_t in_buf [CHUNK * 5];
            uint8_t out_buf[CHUNK * TILE_ROWS * TILE_PSUM_BYTES];
            uint32_t remaining = n;
            while (remaining > 0) {
                const uint32_t batch = remaining < CHUNK ? remaining : CHUNK;
                usb_read_blocking(in_buf, batch * 5);
                uint32_t out_pos = 0;
                for (uint32_t i = 0; i < batch; i++) {
                    const uint8_t flags = in_buf[i * 5 + 0];
                    const uint8_t pw0   = in_buf[i * 5 + 1];
                    const uint8_t pw1   = in_buf[i * 5 + 2];
                    const uint8_t a0    = in_buf[i * 5 + 3];
                    const uint8_t a1    = in_buf[i * 5 + 4];
                    uint8_t * slot = (flags & TILE_FLAG_ENDS_RUN)
                        ? &out_buf[out_pos] : NULL;
                    run_tile_macro(flags, pw0, pw1, a0, a1, slot);
                    if (flags & TILE_FLAG_ENDS_RUN) {
                        out_pos += TILE_ROWS * TILE_PSUM_BYTES;
                    }
                }
                if (out_pos > 0) {
                    usb_write_blocking(out_buf, out_pos);
                }
                remaining -= batch;
            }
            tud_vendor_write_flush();
        }
    }
}
