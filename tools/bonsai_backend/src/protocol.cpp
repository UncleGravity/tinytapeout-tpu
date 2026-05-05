#include "protocol.h"

namespace bonsai {

int build_tile_pairs(uint8_t * dst,
                     const uint8_t * packed_weights,
                     const int8_t * acts,
                     const int16_t * seeds,
                     bool starts_run,
                     bool ends_run,
                     int * out_rdp_byte_offset) {
    int p = 0;
    auto put = [&](uint8_t a, uint8_t b) {
        dst[2 * p]     = a;
        dst[2 * p + 1] = b;
        ++p;
    };

    if (starts_run) {
        put(pack_ui(cmd_clear), 0);
        put(pack_ui(cmd_nop),   0);
    }
    for (int row = 0; row < Tile::rows; ++row) {
        put(pack_ui(cmd_ldw, encode_row_arg(row)), packed_weights[row]);
    }
    for (int lane = 0; lane < Tile::cols; ++lane) {
        put(pack_ui(cmd_lda, encode_col_arg(lane)), (uint8_t) acts[lane]);
    }
    if (starts_run) {
        for (int row = 0; row < Tile::rows; ++row) {
            const uint16_t seed_raw = (uint16_t) seeds[row];
            for (int byte = 0; byte < psum_bytes; ++byte) {
                put(pack_ui(cmd_seed, encode_row_byte_arg(row, byte)),
                    (uint8_t) ((seed_raw >> (byte * 8)) & 0xffu));
            }
        }
    }
    put(pack_ui(cmd_start), 0);
    put(pack_ui(cmd_nop),   0);
    for (int i = 0; i < tile_done_pad_cycles; ++i) put(pack_ui(cmd_nop), 0);

    if (ends_run) {
        if (out_rdp_byte_offset != nullptr) {
            *out_rdp_byte_offset = p;
        }
        for (int row = 0; row < Tile::rows; ++row) {
            for (int byte = 0; byte < psum_bytes; ++byte) {
                put(pack_ui(cmd_rdp, encode_row_byte_arg(row, byte)), 0);
            }
        }
    } else if (out_rdp_byte_offset != nullptr) {
        *out_rdp_byte_offset = -1;
    }

    return p;
}

int16_t parse_psum_at(const uint8_t * rx, size_t rdp_byte_offset, int row) {
    const uint8_t * rdp = rx + rdp_byte_offset + (size_t) row * psum_bytes;
    uint16_t raw = 0;
    for (int byte = 0; byte < psum_bytes; ++byte) {
        raw |= (uint16_t) rdp[byte] << (byte * 8);
    }
    return sign_extend_psum(raw);
}

} // namespace bonsai
