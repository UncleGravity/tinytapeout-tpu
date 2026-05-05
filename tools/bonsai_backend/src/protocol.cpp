#include "protocol.h"

namespace bonsai {

void build_tile_pairs(uint8_t * dst,
                      uint8_t packed_weights,
                      const int8_t * acts,
                      int16_t seed_value) {
    const uint16_t seed_raw = (uint16_t) seed_value;
    int p = 0;
    auto put = [&](uint8_t a, uint8_t b) {
        dst[2 * p]     = a;
        dst[2 * p + 1] = b;
        ++p;
    };
    put(pack_ui(cmd_clear), 0);
    put(pack_ui(cmd_nop),   0);
    put(pack_ui(cmd_ldw, encode_row_arg(0)), packed_weights);
    for (int lane = 0; lane < Tile::cols; ++lane) {
        put(pack_ui(cmd_lda, encode_col_arg(lane)), (uint8_t) acts[lane]);
    }
    for (int byte = 0; byte < psum_bytes; ++byte) {
        put(pack_ui(cmd_seed, encode_row_byte_arg(0, byte)),
            (uint8_t) ((seed_raw >> (byte * 8)) & 0xffu));
    }
    put(pack_ui(cmd_start), 0);
    put(pack_ui(cmd_nop),   0);
    for (int i = 0; i < tile_done_pad_cycles; ++i) put(pack_ui(cmd_nop), 0);
    for (int byte = 0; byte < psum_bytes; ++byte) {
        put(pack_ui(cmd_rdp, encode_row_byte_arg(0, byte)), 0);
    }
}

int16_t parse_tile_psum_at(const uint8_t * rx, int tile_index) {
    const uint8_t * rdp = rx + (size_t) tile_index * tile_cycles + tile_rdp_offset_within_tile;
    uint16_t raw = 0;
    for (int byte = 0; byte < psum_bytes; ++byte) {
        raw |= (uint16_t) rdp[byte] << (byte * 8);
    }
    return sign_extend_psum(raw);
}

} // namespace bonsai
