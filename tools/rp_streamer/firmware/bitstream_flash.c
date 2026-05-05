#include "bitstream_flash.h"
#include "crc32.h"

#include <string.h>

#include "hardware/flash.h"
#include "hardware/sync.h"

// One-sector staging buffer in BSS. 4 KB on a 264 KB-SRAM RP2350 is fine.
// We accumulate up to FLASH_SECTOR_SIZE then flush; the partial tail is
// padded with 0xff (the erased value) inside _end.
static struct {
    uint8_t  buf[FLASH_SECTOR_SIZE];
    uint32_t buf_used;
    uint32_t payload_off;     // bytes of payload written to flash so far
    uint32_t expected_len;
    uint32_t crc;
    bool     active;
} g_state;

bool bitstream_flash_begin(uint32_t total_length) {
    if (total_length == 0 || total_length > BITSTREAM_MAX_BYTES) return false;

    // Erase header page + however many sectors the payload spans, rounded
    // up to the sector boundary. Erase pauses XIP, so we have to disable
    // interrupts (pico-sdk's flash routines are RAM-resident).
    const uint32_t bytes_needed = FLASH_PAGE_SIZE + total_length;
    const uint32_t erase_bytes  = (bytes_needed + FLASH_SECTOR_SIZE - 1)
                                  & ~(FLASH_SECTOR_SIZE - 1);

    const uint32_t ints = save_and_disable_interrupts();
    flash_range_erase(BITSTREAM_FLASH_OFFSET, erase_bytes);
    restore_interrupts(ints);

    g_state.buf_used     = 0;
    g_state.payload_off  = 0;
    g_state.expected_len = total_length;
    g_state.crc          = CRC32_INIT;
    g_state.active       = true;
    return true;
}

bool bitstream_flash_append(const uint8_t * data, size_t n) {
    if (!g_state.active) return false;
    if (g_state.payload_off + g_state.buf_used + n > g_state.expected_len) {
        // Host sent more bytes than it promised in begin().
        g_state.active = false;
        return false;
    }
    g_state.crc = crc32_update(g_state.crc, data, n);

    while (n > 0) {
        size_t copy = FLASH_SECTOR_SIZE - g_state.buf_used;
        if (copy > n) copy = n;
        memcpy(g_state.buf + g_state.buf_used, data, copy);
        g_state.buf_used += copy;
        data += copy;
        n -= copy;

        if (g_state.buf_used == FLASH_SECTOR_SIZE) {
            const uint32_t flash_off = BITSTREAM_FLASH_OFFSET
                                       + FLASH_PAGE_SIZE
                                       + g_state.payload_off;
            const uint32_t ints = save_and_disable_interrupts();
            flash_range_program(flash_off, g_state.buf, FLASH_SECTOR_SIZE);
            restore_interrupts(ints);
            g_state.payload_off += FLASH_SECTOR_SIZE;
            g_state.buf_used = 0;
        }
    }
    return true;
}

int bitstream_flash_end(void) {
    if (!g_state.active) return 1;

    // Flush the partial tail. flash_range_program needs a page-aligned
    // count, so round up and pad with 0xff (the erased state — a no-op
    // overwrite for the unused bytes of the final sector).
    if (g_state.buf_used > 0) {
        const size_t padded = (g_state.buf_used + FLASH_PAGE_SIZE - 1)
                              & ~(FLASH_PAGE_SIZE - 1);
        memset(g_state.buf + g_state.buf_used, 0xff, padded - g_state.buf_used);
        const uint32_t flash_off = BITSTREAM_FLASH_OFFSET
                                   + FLASH_PAGE_SIZE
                                   + g_state.payload_off;
        const uint32_t ints = save_and_disable_interrupts();
        flash_range_program(flash_off, g_state.buf, padded);
        restore_interrupts(ints);
        g_state.payload_off += g_state.buf_used;
        g_state.buf_used = 0;
    }

    if (g_state.payload_off != g_state.expected_len) {
        g_state.active = false;
        return 2;  // host sent fewer bytes than promised
    }

    // Write the header last. Pad to a full page; only the first 16 B are
    // meaningful, the rest stays 0xff (erased state).
    uint8_t header_page[FLASH_PAGE_SIZE];
    memset(header_page, 0xff, sizeof(header_page));
    BitstreamHeader * hdr = (BitstreamHeader *) header_page;
    hdr->magic    = BITSTREAM_MAGIC;
    hdr->length   = g_state.expected_len;
    hdr->crc32    = crc32_finalize(g_state.crc);
    hdr->reserved = 0;

    const uint32_t ints = save_and_disable_interrupts();
    flash_range_program(BITSTREAM_FLASH_OFFSET, header_page, FLASH_PAGE_SIZE);
    restore_interrupts(ints);

    g_state.active = false;

    // Verify by re-reading via XIP.
    if (!bitstream_flash_present() ||
        bitstream_flash_length() != g_state.expected_len) {
        return 3;
    }
    return 0;
}

bool bitstream_flash_present(void) {
    const BitstreamHeader * hdr = (const BitstreamHeader *) BITSTREAM_XIP_BASE;
    if (hdr->magic != BITSTREAM_MAGIC) return false;
    if (hdr->length == 0 || hdr->length > BITSTREAM_MAX_BYTES) return false;
    return true;
}

const uint8_t * bitstream_flash_data(void) {
    return BITSTREAM_XIP_BASE + FLASH_PAGE_SIZE;
}

uint32_t bitstream_flash_length(void) {
    const BitstreamHeader * hdr = (const BitstreamHeader *) BITSTREAM_XIP_BASE;
    return hdr->length;
}
