#pragma once
#include <stddef.h>
#include <stdint.h>

// CRC-32/ISO-HDLC (zlib polynomial 0xEDB88320). Streaming-friendly: pass
// CRC32_INIT into the first call's `crc`, the function's return into the
// next call's `crc`, and finalize with a XOR by 0xffffffff (handled by
// crc32_finalize).
#define CRC32_INIT  0xffffffffu

uint32_t crc32_update(uint32_t crc, const uint8_t * data, size_t n);

static inline uint32_t crc32_finalize(uint32_t crc) {
    return crc ^ 0xffffffffu;
}
