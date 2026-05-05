#pragma once

#include <cstddef>
#include <cstdint>

// Single source of truth for the chip-cycle protocol.
//
// Mirrors the RTL pin protocol. Every consumer of the protocol
// reads from this header.

namespace bonsai {

// ---------------------------------------------------------------------------
// Tile geometry. Must match RTL parameters.
struct Tile {
    static constexpr int rows      = 2;
    static constexpr int cols      = 2;
    static constexpr int act_bits  = 8;
    static constexpr int psum_bits = 16;
};

// ---------------------------------------------------------------------------
// Status byte returned by CMD_STATUS / CMD_NOP.
enum Status : uint8_t {
    status_busy        = 1u << 0,
    status_done        = 1u << 1,
    status_weight_done = 1u << 2,
    status_all_valid   = 1u << 3,
    status_start_ready = 1u << 4,
    status_idle_stable = 1u << 5,
    status_error       = 1u << 6,
};

// ---------------------------------------------------------------------------
// Command opcodes. Sit in ui_in[2:0] (ui_in[7:3] is the 5-bit arg field).
// See tpu_cmd_decode.v.
enum Command : uint8_t {
    cmd_status = 0,
    cmd_clear  = 1,
    cmd_ldw    = 2,
    cmd_lda    = 3,
    cmd_seed   = 4,
    cmd_start  = 5,
    cmd_rdp    = 6,
    cmd_nop    = 7,
};

// ---------------------------------------------------------------------------
// Argument-field layout helpers.
//
// arg = ui_in[7:3] is 5 bits. Different commands carve it up differently:
//   LDW:   arg = row
//   LDA:   arg = col
//   SEED:  arg = row | (byte << row_bits)
//   RDP:   arg = row | (byte << row_bits)

constexpr int bit_width_for_count(int count) {
    int bits = 1;
    int max_value = count - 1;
    while ((1 << bits) <= max_value) {
        ++bits;
    }
    return bits;
}

constexpr int row_bits   = bit_width_for_count(Tile::rows);
constexpr int col_bits   = bit_width_for_count(Tile::cols);
constexpr int psum_bytes = (Tile::psum_bits + 7) / 8;

static_assert(Tile::cols <= 8, "LDW packs one tile row into an 8-bit data byte");

inline uint8_t pack_ui(Command cmd, uint8_t arg = 0) {
    return (uint8_t) cmd | (uint8_t) ((arg & 0x1fu) << 3);
}

inline uint8_t encode_row_arg(int row) {
    return (uint8_t) (row & ((1 << row_bits) - 1));
}

inline uint8_t encode_col_arg(int col) {
    return (uint8_t) (col & ((1 << col_bits) - 1));
}

inline uint8_t encode_row_byte_arg(int row, int byte) {
    return (uint8_t) (encode_row_arg(row) | (byte << row_bits));
}

inline int16_t sign_extend_psum(uint16_t raw) {
    constexpr uint16_t sign = 1u << (Tile::psum_bits - 1);
    constexpr uint16_t mask = (Tile::psum_bits == 16) ? 0xffffu : ((1u << Tile::psum_bits) - 1u);
    raw &= mask;
    return (raw & sign) ? (int16_t) ((int32_t) raw - (int32_t) (mask + 1u)) : (int16_t) raw;
}

// ---------------------------------------------------------------------------
// Per-tile X-frame body layout. A "tile" is one chip-side compute unit:
//   clear+nop / ldw / lda × cols / seed × psum_bytes / start+nop /
//   nop × tile_done_pad_cycles / rdp × psum_bytes
//
// `tile_done_pad_cycles` covers the chip's compute latency from start_pulse
// to acc_q being valid for read-back.

constexpr int tile_done_pad_cycles = 6;
constexpr int tile_cycles =
    /* clear+nop  */ 2 +
    /* ldw        */ 1 +
    /* lda × cols */ Tile::cols +
    /* seed bytes */ psum_bytes +
    /* start+nop  */ 2 +
    /* done pad   */ tile_done_pad_cycles +
    /* rdp bytes  */ psum_bytes;
constexpr int tile_rdp_offset_within_tile = tile_cycles - psum_bytes;

// Fill `dst[2 * tile_cycles]` with one tile's (ui, uio) byte pairs. Caller
// wraps N tiles in a single 5-byte X-frame header.
void build_tile_pairs(uint8_t * dst,
                      uint8_t packed_weights,
                      const int8_t * acts,
                      int16_t seed_value);

// Read the row-0 psum out of `rx[tile_index * tile_cycles ..]`. `rx` is the
// raw uo_out byte stream returned by one X-frame.
int16_t parse_tile_psum_at(const uint8_t * rx, int tile_index);

} // namespace bonsai
