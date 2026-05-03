#include <stdint.h>
#include "pico/stdlib.h"
#include "pico/bootrom.h"
#include "hardware/gpio.h"
#include "hardware/structs/sio.h"
#include "tusb.h"

// Protocol: host sends 5-byte header [mode][n_le_u32] per request.
//
// Throughput modes (no chip pin activity):
//   'T' -> RP transmits N bytes
//   'R' -> RP receives and drops N bytes
//   'E' -> RP echoes N bytes
//   'B' -> RP reboots into USB BOOTSEL (n ignored)
//
// Chip-control modes (drive the TT pins):
//   'a' -> assert chip reset (rst_n LOW), n ignored
//   'd' -> deassert chip reset (rst_n HIGH), n ignored
//   'X' -> stream N transactions. Body: 2*N bytes [ui_in, uio_in]+. Reply: N
//          bytes (uo_out sampled after each clock edge).
//
// Pin map (ETR demoboard):
//   rst_n      = GPIO14                 (bank 0)
//   clock      = GPIO16                 (bank 0)
//   ui_in[0:7] = GPIO17:24              (bank 0)
//   uio[0:6]   = GPIO25:31              (bank 0)
//   uio[7]     = GPIO32                 (bank 1!)
//   uo_out[0:7]= GPIO33:40              (bank 1 -- gpio_hi_in on RP2350)
//
// RP2350 SIO splits GPIOs 0-31 (gpio_out/in) from 32-47 (gpio_hi_out/in), so
// uio[7] has to be driven via gpio_hi_set / gpio_hi_clr, not gpio_out. A
// single 16-bit shifted write to gpio_out silently truncates bit 32.

#define PIN_RESET    14
#define PIN_CLOCK    16
#define PIN_UI_BASE  17
#define PIN_UIO_BASE 25
#define PIN_UO_BASE  33

// 15 bits of {ui_in[0:7], uio[0:6]} live in bank 0 at positions 17..31.
#define UI_UIO_BANK0_MASK  ((uint32_t) 0x7FFFu << PIN_UI_BASE)
// uio[7] is GPIO32, i.e. bit 0 of bank 1.
#define UIO7_BANK1_BIT     (1u << (PIN_UIO_BASE + 7 - 32))

static void chip_pins_init(void) {
    for (int g = PIN_UI_BASE; g < PIN_UI_BASE + 16; g++) {
        gpio_init(g);
        gpio_set_dir(g, GPIO_OUT);
        gpio_put(g, 0);
    }
    gpio_init(PIN_CLOCK);
    gpio_set_dir(PIN_CLOCK, GPIO_OUT);
    gpio_put(PIN_CLOCK, 0);

    gpio_init(PIN_RESET);
    gpio_set_dir(PIN_RESET, GPIO_OUT);
    gpio_put(PIN_RESET, 1); // start out of reset; host can re-assert

    for (int g = PIN_UO_BASE; g < PIN_UO_BASE + 8; g++) {
        gpio_init(g);
        gpio_set_dir(g, GPIO_IN);
    }
}

// Spin for ~n CPU cycles to give the iCE40 + pad round-trip and the RP2350
// input synchronizer time to settle.
static inline void delay_cycles(uint32_t n) {
    asm volatile (
        "1: subs %0, %0, #1\n"
        "   bne 1b\n"
        : "+r" (n)
        :
        : "cc"
    );
}

// On the FPGA breakout the on-chip path is roughly:
//   ui_in pad -> iCE40 routing -> comb mux (RDP) or status reg -> uo_out pad
//   -> RP2350 GPIO input (2-flop synchronizer)
// At 150 MHz CPU, 32 cycles ~= 213 ns of headroom each side. This is well
// inside the FS-USB ceiling so the slack is free.
#define CHIP_SETUP_CYCLES   32
#define CHIP_HOLD_CYCLES    32

static inline uint8_t chip_cycle(uint8_t ui, uint8_t uio) {
    // Bank 0: 8 bits of ui_in at GPIO17, 7 LSBs of uio at GPIO25.
    uint32_t bank0_value = ((uint32_t) ui << PIN_UI_BASE)
                         | ((uint32_t) (uio & 0x7Fu) << PIN_UIO_BASE);
    sio_hw->gpio_out = (sio_hw->gpio_out & ~UI_UIO_BANK0_MASK) | bank0_value;
    // Bank 1: uio[7] -> GPIO32.
    if (uio & 0x80u) sio_hw->gpio_hi_set = UIO7_BANK1_BIT;
    else             sio_hw->gpio_hi_clr = UIO7_BANK1_BIT;

    delay_cycles(CHIP_SETUP_CYCLES);
    // Rising edge.
    sio_hw->gpio_set = 1u << PIN_CLOCK;
    delay_cycles(CHIP_HOLD_CYCLES);
    // Sample uo_out from bank 1: gpio_hi_in bit i = GPIO (32+i).
    uint32_t hi = sio_hw->gpio_hi_in;
    uint8_t uo = (uint8_t) ((hi >> (PIN_UO_BASE - 32)) & 0xFFu);
    // Falling edge.
    sio_hw->gpio_clr = 1u << PIN_CLOCK;
    return uo;
}

static void cdc_read_blocking(uint8_t *p, uint32_t n) {
    uint32_t got = 0;
    while (got < n) {
        tud_task();
        if (tud_cdc_available()) {
            got += tud_cdc_read(p + got, n - got);
        }
    }
}

static void cdc_write_blocking(const uint8_t *p, uint32_t n) {
    uint32_t put = 0;
    while (put < n) {
        tud_task();
        uint32_t w = tud_cdc_write(p + put, n - put);
        if (w) put += w;
        else tud_cdc_write_flush();
    }
}

int main(void) {
    tud_init(0);
    chip_pins_init();

    static uint8_t buf[4096];
    for (uint32_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t) i;

    while (1) {
        uint8_t hdr[5];
        cdc_read_blocking(hdr, sizeof(hdr));
        uint8_t mode = hdr[0];
        uint32_t n = (uint32_t) hdr[1]
                   | ((uint32_t) hdr[2] << 8)
                   | ((uint32_t) hdr[3] << 16)
                   | ((uint32_t) hdr[4] << 24);

        if (mode == 'B') {
            tud_cdc_write_flush();
            sleep_ms(50);
            reset_usb_boot(0, 0);
        } else if (mode == 'T') {
            uint32_t sent = 0;
            while (sent < n) {
                uint32_t want = n - sent;
                if (want > sizeof(buf)) want = sizeof(buf);
                cdc_write_blocking(buf, want);
                sent += want;
            }
            tud_cdc_write_flush();
        } else if (mode == 'R') {
            uint32_t got = 0;
            while (got < n) {
                tud_task();
                if (tud_cdc_available()) {
                    uint32_t want = n - got;
                    if (want > sizeof(buf)) want = sizeof(buf);
                    got += tud_cdc_read(buf, want);
                }
            }
        } else if (mode == 'E') {
            uint32_t got = 0;
            while (got < n) {
                tud_task();
                if (tud_cdc_available()) {
                    uint32_t want = n - got;
                    if (want > sizeof(buf)) want = sizeof(buf);
                    uint32_t r = tud_cdc_read(buf, want);
                    cdc_write_blocking(buf, r);
                    got += r;
                }
            }
            tud_cdc_write_flush();
        } else if (mode == 'a') {
            sio_hw->gpio_clr = 1u << PIN_RESET;
        } else if (mode == 'd') {
            sio_hw->gpio_set = 1u << PIN_RESET;
        } else if (mode == 'X') {
            // Stream chip cycles synchronously, 2 bytes in -> 1 byte out.
            uint8_t pair[2];
            for (uint32_t i = 0; i < n; i++) {
                cdc_read_blocking(pair, 2);
                uint8_t uo = chip_cycle(pair[0], pair[1]);
                cdc_write_blocking(&uo, 1);
            }
            tud_cdc_write_flush();
        }
    }
}
