#include "ice40_loader.h"

#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/spi.h"

// Hardware SPI on spi0:
//   GPIO 3 (MOSI) is spi0 TX  on alt-function 1
//   GPIO 6 (SCK)  is spi0 SCK on alt-function 1
// SS and CRESET_B stay as plain SIO outputs because the iCE40 slave-SPI
// init handshake requires us to time SS edges manually around the
// 8-dummy-clock burst that gates entry into config mode (see TN1248
// "iCE40 Programming and Configuration").
#define ICE40_SPI_HW    spi0
#define ICE40_SPI_HZ    1000000  // 1 MHz, matches TT-MP default

void ice40_pins_init(void) {
    gpio_init(ICE40_PIN_RESET);
    gpio_set_dir(ICE40_PIN_RESET, GPIO_OUT);
    gpio_put(ICE40_PIN_RESET, 1);  // de-asserted at boot

    gpio_init(ICE40_PIN_SS);
    gpio_set_dir(ICE40_PIN_SS, GPIO_OUT);
    gpio_put(ICE40_PIN_SS, 1);     // de-asserted at boot
}

bool ice40_load_bitstream(const uint8_t * data, size_t n) {
    if (n == 0) return false;

    // Initialize SPI and route SCK/MOSI to the peripheral. We deinit at
    // the end so the pins return to plain SIO (and could be repurposed).
    spi_init(ICE40_SPI_HW, ICE40_SPI_HZ);
    spi_set_format(ICE40_SPI_HW,
                   /*data_bits=*/ 8,
                   /*cpol=*/ SPI_CPOL_0,
                   /*cpha=*/ SPI_CPHA_0,
                   /*order=*/ SPI_MSB_FIRST);
    gpio_set_function(ICE40_PIN_SCK,  GPIO_FUNC_SPI);
    gpio_set_function(ICE40_PIN_MOSI, GPIO_FUNC_SPI);

    // === Sequence translated verbatim from TT-MP spi_transferPIO ===

    // 1. Assert reset, hold SS low.
    gpio_put(ICE40_PIN_RESET, 0);
    gpio_put(ICE40_PIN_SS,    0);
    sleep_us(15000);  // generous; datasheet minimum is ~200 ns

    // 2. Release reset, wait for the iCE40 to be ready for config.
    gpio_put(ICE40_PIN_RESET, 1);
    sleep_us(15000);  // datasheet "wake-up" window: minimum 1200 us

    // 3. Pull SS high and send 8 dummy clocks; then re-assert SS. This
    //    is the iCE40 slave-SPI init handshake — 8 clocks while SS is
    //    high gates the FPGA into accepting the bitstream.
    gpio_put(ICE40_PIN_SS, 1);
    sleep_us(2000);
    {
        const uint8_t zero = 0;
        spi_write_blocking(ICE40_SPI_HW, &zero, 1);
    }
    gpio_put(ICE40_PIN_SS, 0);
    sleep_us(2000);

    // 4. Stream the bitstream.
    spi_write_blocking(ICE40_SPI_HW, data, n);

    // 5. Send 6 termination bytes (48 trailing clocks; iCE40 needs >= 49
    //    cycles after the last bitstream bit to finish entering user mode).
    {
        static const uint8_t term[6] = {0};
        spi_write_blocking(ICE40_SPI_HW, term, sizeof(term));
    }

    // 6. Deselect.
    gpio_put(ICE40_PIN_SS, 1);

    // 7. Tear down SPI and return SCK/MOSI to SIO so the pins can be
    //    used (or simply not driven) by other code.
    gpio_set_function(ICE40_PIN_SCK,  GPIO_FUNC_SIO);
    gpio_set_function(ICE40_PIN_MOSI, GPIO_FUNC_SIO);
    spi_deinit(ICE40_SPI_HW);

    // CDONE is not wired into TT-MP's reference impl, so we don't read
    // it back either. The chip itself will fail the next status() if
    // configuration didn't take.
    return true;
}
