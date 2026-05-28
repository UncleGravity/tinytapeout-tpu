/*
 * q1a8_layout.c — pack/merge for the canonical Q1A8 wire format.
 *
 * Linked into both libggml-pynq.so (host backend, called from
 * pynq_buffer_set_tensor at upload time) and libbonsai_ps.so (board
 * runtime, called from the PL matmul driver to merge per-column acts
 * into pre-packed weights). See the doc-comment in q1a8_layout.h for
 * the wire format.
 *
 * Designed for memcpy throughput on ARM Cortex-A9, not generality: the
 * inner loops do 8 / 32 / 32-byte memcpys and 64-bit stores only. No
 * bit shuffles in the hot path (those were already paid at pack_weights
 * time, which runs once per model upload).
 */

#include "q1a8_layout.h"

#include <string.h>

static inline uint16_t read_le_u16(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static inline uint32_t read_le_u32(const uint8_t * p) {
    return (uint32_t) p[0]
         | ((uint32_t) p[1] << 8)
         | ((uint32_t) p[2] << 16)
         | ((uint32_t) p[3] << 24);
}

static inline void write_le_u64(uint8_t * out, uint64_t value) {
    out[0] = (uint8_t) (value);
    out[1] = (uint8_t) (value >> 8);
    out[2] = (uint8_t) (value >> 16);
    out[3] = (uint8_t) (value >> 24);
    out[4] = (uint8_t) (value >> 32);
    out[5] = (uint8_t) (value >> 40);
    out[6] = (uint8_t) (value >> 48);
    out[7] = (uint8_t) (value >> 56);
}

int bonsai_q1a8_pack_weights(
    const uint8_t * q1_0_weights,
    uint32_t rows,
    uint32_t k,
    uint8_t * out) {
    if (q1_0_weights == NULL || out == NULL) return -1;
    if (k == 0 || (k % BONSAI_Q1_BLOCK) != 0) return -2;
    if (rows == 0) return -3;

    const uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    const uint32_t weight_row_bytes = blocks_per_row * BONSAI_Q1_BLOCK_BYTES;
    const uint32_t rowblocks =
        (rows + BONSAI_Q1A8_ROWS_PER_BLOCK - 1) / BONSAI_Q1A8_ROWS_PER_BLOCK;

    size_t cursor = 0;

    for (uint32_t rb = 0; rb < rowblocks; ++rb) {
        const uint32_t row_start = rb * BONSAI_Q1A8_ROWS_PER_BLOCK;
        uint32_t row_count = rows - row_start;
        if (row_count > BONSAI_Q1A8_ROWS_PER_BLOCK) {
            row_count = BONSAI_Q1A8_ROWS_PER_BLOCK;
        }

        for (uint32_t q1 = 0; q1 < blocks_per_row; ++q1) {
            /* weight_scales: SCALE_BEATS beats of 4 fp16 each. */
            for (uint32_t beat = 0; beat < BONSAI_Q1A8_SCALE_BEATS; ++beat) {
                uint64_t word = 0;
                for (uint32_t local = 0; local < 4; ++local) {
                    const uint32_t lane = beat * 4 + local;
                    uint16_t scale = 0;
                    if (lane < row_count) {
                        const size_t off =
                            (size_t) (row_start + lane) * weight_row_bytes
                            + (size_t) q1 * BONSAI_Q1_BLOCK_BYTES;
                        scale = read_le_u16(q1_0_weights + off);
                    }
                    word |= ((uint64_t) scale) << (local * 16);
                }
                write_le_u64(out + cursor, word);
                cursor += 8;
            }

            /* wbits: 4 sub-blocks of WBITS_BEATS each (2 u32 per beat). */
            for (uint32_t sub = 0; sub < BONSAI_Q1A8_Q8_SUBBLOCKS; ++sub) {
                for (uint32_t beat = 0; beat < BONSAI_Q1A8_WBITS_BEATS; ++beat) {
                    uint64_t word = 0;
                    for (uint32_t local = 0; local < 2; ++local) {
                        const uint32_t lane = beat * 2 + local;
                        uint32_t bits = 0;
                        if (lane < row_count) {
                            const size_t base =
                                (size_t) (row_start + lane) * weight_row_bytes
                                + (size_t) q1 * BONSAI_Q1_BLOCK_BYTES
                                + 2  /* skip fp16 scale */
                                + (size_t) sub * (BONSAI_Q8_BLOCK / 8);
                            bits = read_le_u32(q1_0_weights + base);
                        }
                        word |= ((uint64_t) bits) << (local * 32);
                    }
                    write_le_u64(out + cursor, word);
                    cursor += 8;
                }
            }
        }
    }

    return 0;
}

int bonsai_q1a8_pack_acts(
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint32_t k,
    uint8_t * out) {
    if (act_quants == NULL || act_scale_bits == NULL || out == NULL) return -1;
    if (k == 0 || (k % BONSAI_Q1_BLOCK) != 0) return -2;

    const uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    uint8_t * s = out;

    /* Layout matches the v4 kernel's LOAD_ACTS FSM: per Q1 block, 4 sub-
     * blocks each = (32 B acts) + (8 B fp16 scale in low 16 bits). */
    for (uint32_t q1 = 0; q1 < blocks_per_row; ++q1) {
        for (uint32_t sub = 0; sub < BONSAI_Q1A8_Q8_SUBBLOCKS; ++sub) {
            const uint32_t q8_idx = q1 * BONSAI_Q1A8_Q8_SUBBLOCKS + sub;
            memcpy(s, act_quants + (size_t) q8_idx * BONSAI_Q8_BLOCK,
                BONSAI_Q1A8_ACTS_BYTES);
            s += BONSAI_Q1A8_ACTS_BYTES;
            write_le_u64(s, (uint64_t) act_scale_bits[q8_idx]);
            s += BONSAI_Q1A8_ACT_SCALE_BYTES;
        }
    }
    return 0;
}

int bonsai_q1a8_merge_acts(
    const uint8_t * packed_weights,
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint32_t rows,
    uint32_t k,
    uint8_t * out_stream) {
    if (packed_weights == NULL || act_quants == NULL ||
        act_scale_bits == NULL || out_stream == NULL) return -1;
    if (k == 0 || (k % BONSAI_Q1_BLOCK) != 0) return -2;
    if (rows == 0) return -3;

    const uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    const uint32_t rowblocks =
        (rows + BONSAI_Q1A8_ROWS_PER_BLOCK - 1) / BONSAI_Q1A8_ROWS_PER_BLOCK;

    const uint8_t * p = packed_weights;
    uint8_t * s = out_stream;

    for (uint32_t rb = 0; rb < rowblocks; ++rb) {
        for (uint32_t q1 = 0; q1 < blocks_per_row; ++q1) {
            /* weight_scales: direct copy. */
            memcpy(s, p, BONSAI_Q1A8_SCALES_BYTES);
            s += BONSAI_Q1A8_SCALES_BYTES;
            p += BONSAI_Q1A8_SCALES_BYTES;

            for (uint32_t sub = 0; sub < BONSAI_Q1A8_Q8_SUBBLOCKS; ++sub) {
                /* acts: 32 int8, broadcast across all rows in this rowblock. */
                const uint32_t a_off =
                    q1 * BONSAI_Q1_BLOCK + sub * BONSAI_Q8_BLOCK;
                memcpy(s, act_quants + a_off, BONSAI_Q1A8_ACTS_BYTES);
                s += BONSAI_Q1A8_ACTS_BYTES;

                /* act_scale: fp16 in the low 16 bits of an 8-byte beat. */
                const uint32_t scale_index =
                    (q1 * BONSAI_Q1_BLOCK + sub * BONSAI_Q8_BLOCK) / BONSAI_Q8_BLOCK;
                write_le_u64(s, (uint64_t) act_scale_bits[scale_index]);
                s += BONSAI_Q1A8_ACT_SCALE_BYTES;

                /* wbits: direct copy. */
                memcpy(s, p, BONSAI_Q1A8_WBITS_BYTES);
                s += BONSAI_Q1A8_WBITS_BYTES;
                p += BONSAI_Q1A8_WBITS_BYTES;
            }
        }
    }

    return 0;
}
