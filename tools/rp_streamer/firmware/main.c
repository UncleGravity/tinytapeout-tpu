#include <stdint.h>
#include "pico/stdlib.h"
#include "pico/bootrom.h"
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
//   'F' -> stream N bytes of iCE40 bitstream into RP2350 flash; on
//          completion, also re-loads it into the FPGA. Reply: 1 byte
//          status (0=ok, nonzero=error code from bitstream_flash_end).
//   'L' -> re-load the bitstream currently stored in flash into the FPGA.
//          Reply: 1 byte (0=ok, 1=no bitstream stored, 2=load failed).
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
        }
    }
}
