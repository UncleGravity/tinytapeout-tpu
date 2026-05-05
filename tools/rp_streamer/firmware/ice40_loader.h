#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// FabricFox v2 iCE40 slave-SPI bitstream loader.
//
// Pin assignments are copied verbatim from TinyTapeout/tt-micropython-firmware
// `src/ttboard/fpga/fabricfoxv2.py` + `gpio_map_dbv3.py` (GPIOMapTTDBv3).
// Verified to not collide with the chip-pin map in main.c.

#define ICE40_PIN_RESET  1   // CRESET_B   (CTRL_SEL_nRST in TT-MP)
#define ICE40_PIN_MOSI   3   // SDO        (MNG00 in TT-MP)
#define ICE40_PIN_SS     5   // chip-select (MNG02 in TT-MP)
#define ICE40_PIN_SCK    6   // SCK        (MNG03 in TT-MP)

void ice40_pins_init(void);

// Configure iCE40 in slave-SPI mode and stream the bitstream.
// Returns true on success. Mirrors the TT-MP spi_transferPIO sequence.
bool ice40_load_bitstream(const uint8_t * data, size_t n);
