#include "crc32.h"

// Slice-by-1 table-driven CRC32. ~5 cycles/byte, 1 KB table in flash.
// Generated lazily on first call so we don't pay flash for the constants
// when the bitstream loader isn't used.

static uint32_t table[256];
static int table_ready = 0;

static void build_table(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    table_ready = 1;
}

uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t n) {
    if (!table_ready) build_table();
    for (size_t i = 0; i < n; ++i) {
        crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
    }
    return crc;
}
