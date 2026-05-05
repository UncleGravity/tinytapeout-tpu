#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "hardware/flash.h"
#include "pico/binary_info.h"

// On-flash bitstream storage in the top 256 KB of XIP flash.
//
// Layout inside the region:
//   [0 .. FLASH_PAGE_SIZE)            — BitstreamHeader (padded to one page)
//   [FLASH_PAGE_SIZE .. region end)    — bitstream payload
//
// Updated via the streaming begin/append/end API below; the host CLI
// pushes bytes through the firmware's 'F' mode, which calls these.

#define BITSTREAM_FLASH_REGION_BYTES (256u * 1024u)
#define BITSTREAM_FLASH_OFFSET       (PICO_FLASH_SIZE_BYTES - BITSTREAM_FLASH_REGION_BYTES)
#define BITSTREAM_XIP_BASE           ((const uint8_t *) (XIP_BASE + BITSTREAM_FLASH_OFFSET))
#define BITSTREAM_MAX_BYTES          (BITSTREAM_FLASH_REGION_BYTES - FLASH_PAGE_SIZE)

#define BITSTREAM_MAGIC 0x42535442u  // "BTSB" little-endian

typedef struct {
    uint32_t magic;     // BITSTREAM_MAGIC if region holds a valid bitstream
    uint32_t length;    // payload bytes that follow
    uint32_t crc32;     // CRC-32/ISO-HDLC of the payload
    uint32_t reserved;  // pad to 16 B
} BitstreamHeader;

// Streaming write. Pattern:
//   bitstream_flash_begin(total_length)   — erases the region
//   bitstream_flash_append(buf, n)        — accumulate; flushes per sector
//   bitstream_flash_end()                 — flush partial + write header
//
// Concurrency: all three pause XIP via save_and_disable_interrupts().
// Caller must not be holding any other flash-resident pointer that it
// needs during the call window.
bool bitstream_flash_begin(uint32_t total_length);
bool bitstream_flash_append(const uint8_t * data, size_t n);
int  bitstream_flash_end(void);  // 0 = ok, nonzero = error code

// Boot-side accessors.
bool             bitstream_flash_present(void);
const uint8_t *  bitstream_flash_data(void);
uint32_t         bitstream_flash_length(void);
