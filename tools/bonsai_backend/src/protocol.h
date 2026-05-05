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
// Per-tile X-frame body layout. Both array rows are always exercised
// (uniform "fire shape" on the wire — see plan.h). A tile carries optional
// CLEAR/SEED head and RDP tail, gated by its run flags:
//
//   [clear+nop]                 — only when starts_run
//   ldw × ROWS
//   lda × COLS
//   [seed × ROWS × psum_bytes]  — only when starts_run
//   start+nop
//   nop × tile_done_pad_cycles
//   [rdp  × ROWS × psum_bytes]  — only when ends_run
//
// Within a multi-tile run, intermediate tiles drop CLEAR/SEED/RDP — the
// chip's acc_q persists across consecutive STARTs and accumulates each
// fire's contribution. The acc_mem RTL preserves acc_q on start_pulse;
// CLEAR (or SEED) at the run head re-zeros it.
//
// `tile_done_pad_cycles` covers the chip's compute latency from start_pulse
// back to IDLE. RDP bytes are emitted row-major when present:
// [row0_byte0, row0_byte1, row1_byte0, row1_byte1].

constexpr int tile_done_pad_cycles = 6;

// Cycle count at each phase, conditional on run flags.
constexpr int tile_head_cycles_full     = 2 + Tile::rows * psum_bytes;  // clear+nop + seed
constexpr int tile_head_cycles_skip     = 0;
constexpr int tile_body_cycles          = Tile::rows + Tile::cols + 2 + tile_done_pad_cycles;
constexpr int tile_tail_cycles_full     = Tile::rows * psum_bytes;      // rdp
constexpr int tile_tail_cycles_skip     = 0;

constexpr int tile_cycles_for(bool starts_run, bool ends_run) {
    return (starts_run ? tile_head_cycles_full : tile_head_cycles_skip)
         + tile_body_cycles
         + (ends_run   ? tile_tail_cycles_full : tile_tail_cycles_skip);
}

// Worst-case (standalone) tile cycle count, kept around so callers that
// allocate a fixed-stride scratch buffer still have an upper bound.
constexpr int tile_cycles_max = tile_cycles_for(true, true);

// Fill `dst` with one tile's (ui, uio) byte pairs. `dst` must have room for
// `2 * tile_cycles_for(starts_run, ends_run)` bytes. Returns the number of
// chip cycles emitted (== tile_cycles_for(...)). RDP bytes (when present)
// land at the end of the emitted span; `out_rdp_byte_offset` returns the
// offset (in uo_out bytes from the start of this tile) of the first RDP
// result byte, or -1 if the tile doesn't end a run.
int build_tile_pairs(uint8_t * dst,
                     const uint8_t * packed_weights,   // [Tile::rows]
                     const int8_t * acts,              // [Tile::cols]
                     const int16_t * seeds,            // [Tile::rows]
                     bool starts_run,
                     bool ends_run,
                     int * out_rdp_byte_offset);

// Read row `row` (0..Tile::rows-1) out of the rx byte stream at the given
// absolute offset (returned by build_tile_pairs).
int16_t parse_psum_at(const uint8_t * rx, size_t rdp_byte_offset, int row);

} // namespace bonsai
