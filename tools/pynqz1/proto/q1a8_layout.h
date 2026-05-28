/*
 * q1a8_layout.h — canonical wire formats for the Q1A8 rowblock kernel.
 *
 * As of bitstream v4 the PL kernel has TWO AXIS slave ports:
 *
 *   S_AXIS         weights stream (= the packed_weights bytes verbatim)
 *   S_AXIS_ACTS    acts+scales stream, sent once per matmul column
 *
 * The kernel loads the acts stream into a small BRAM at the start of each
 * column, then iterates rowblocks reading weight bits/scales from the
 * weights stream and acts/scales from BRAM. Net: no per-rowblock act
 * repetition (~2× less DMA traffic), and the weights stream is exactly
 * the on-storage packed_weights with zero per-matmul transformation.
 *
 * --- v1 historical (interleaved) ---
 *
 * The pre-v4 single-stream format interleaved everything per Q1 block per
 * rowblock (304 bytes). Kept here for reference and used by the legacy
 * bonsai_pack_matmul_q1a8_stream which only the e2e parity test consumes.
 *
 *   [ 16 B  weight_scales       ]  2 beats × 4 fp16 scales each (rows 0..7)
 *   [ 32 B  acts  sub-block 0   ]  4 beats × 32 int8 (broadcast across rows)
 *   [  8 B  act_scale sub-block 0]  1 beat,  fp16 in low 16 bits
 *   [ 32 B  wbits sub-block 0   ]  4 beats × 2 u32 (one wbits word per row)
 *   [ 32 B  acts  sub-block 1   ]
 *   [  8 B  act_scale sub-block 1]
 *   [ 32 B  wbits sub-block 1   ]
 *   [ ... sub-blocks 2 and 3 ... ]
 *   total: 304 bytes per Q1 block per rowblock
 *
 * --- v4 weights stream (per Q1 block per rowblock = 144 B) ---
 *
 *   [ 16 B  weight_scales       ]  2 beats × 4 fp16 each
 *   [ 32 B  wbits sub-block 0   ]  4 beats × 2 u32 each
 *   [ 32 B  wbits sub-block 1   ]
 *   [ 32 B  wbits sub-block 2   ]
 *   [ 32 B  wbits sub-block 3   ]
 *   total: 144 B (= 8 × Q1_BLOCK_BYTES — same as Q1_0 source for 8 rows)
 *
 * Bit-identical to the on-DDR packed_weights layout, so the driver DMAs
 * packed_weights straight from CMA with no transformation.
 *
 * --- v4 acts stream (per matmul column, sent once) ---
 *
 *   For each Q1 block in K:
 *     [ 32 B  acts sub-block 0    ]  4 beats × 8 bytes int8
 *     [  8 B  act_scale sub-block 0]  fp16 in low 16 bits
 *     [ 32 B  acts sub-block 1    ]
 *     [  8 B  act_scale sub-block 1]
 *     [ ... sub-blocks 2 and 3 ... ]
 *   total: (k / Q1_BLOCK) × 4 × 40  =  k × 1.25 bytes per column
 *   (k=2048 → 2560 bytes per column)
 *
 * The PACKED layout (what the host stores on the board, the runtime merge
 * function reads from) carries only the weight portion of that:
 *
 *   [ 16 B  weight_scales       ]
 *   [ 32 B  wbits sub-block 0   ]
 *   [ 32 B  wbits sub-block 1   ]
 *   [ 32 B  wbits sub-block 2   ]
 *   [ 32 B  wbits sub-block 3   ]
 *   total: 144 bytes per Q1 block per rowblock
 *
 * which is *exactly* 8 × Q1_BLOCK_BYTES (8 rows × 18). So Q1_0 weight
 * tensors take the same total bytes on the board after repacking — only
 * the byte order changes.
 *
 * The outer order in both layouts is rowblock-major then q1_block-major:
 *   packed[rb * blocks_per_row + q1] = 144 bytes
 *   stream[rb * blocks_per_row + q1] = 304 bytes
 *
 * Keep in sync with proto/q1a8_layout.py and the Verilog FSM in
 * fpga/rtl/q1a8/q1a8_kernel.v.
 */

#ifndef PYNQ_PROTO_Q1A8_LAYOUT_H
#define PYNQ_PROTO_Q1A8_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block sizes shared with proto/ops.{h,py}. Duplicated here so this header
 * is self-contained; values must match. */
#define BONSAI_Q1_BLOCK              128
#define BONSAI_Q1_BLOCK_BYTES         18   /* fp16 scale + 16 bytes wbits */
#define BONSAI_Q8_BLOCK               32
#define BONSAI_Q1A8_ROWS_PER_BLOCK     8

/* Derived beat counts. Computed from ROWS_PER_BLOCK; kept as macros so
 * Verilog and C agree without a config file. */
#define BONSAI_Q1A8_SCALE_BEATS    ((BONSAI_Q1A8_ROWS_PER_BLOCK + 3) / 4)
#define BONSAI_Q1A8_WBITS_BEATS    ((BONSAI_Q1A8_ROWS_PER_BLOCK + 1) / 2)
#define BONSAI_Q1A8_Q8_SUBBLOCKS   (BONSAI_Q1_BLOCK / BONSAI_Q8_BLOCK)  /* 4 */

/* Byte counts per Q1 block per rowblock. */
#define BONSAI_Q1A8_SCALES_BYTES   (BONSAI_Q1A8_SCALE_BEATS * 8)            /* 16 */
#define BONSAI_Q1A8_ACTS_BYTES     (BONSAI_Q8_BLOCK)                        /* 32 */
#define BONSAI_Q1A8_ACT_SCALE_BYTES 8
#define BONSAI_Q1A8_WBITS_BYTES    (BONSAI_Q1A8_WBITS_BEATS * 8)            /* 32 */
#define BONSAI_Q1A8_SUBBLOCK_STREAM_BYTES \
    (BONSAI_Q1A8_ACTS_BYTES + BONSAI_Q1A8_ACT_SCALE_BYTES + BONSAI_Q1A8_WBITS_BYTES)
/* Legacy v1 interleaved stream: 304 B per Q1 block per rowblock. */
#define BONSAI_Q1A8_STREAM_PER_Q1_BLOCK \
    (BONSAI_Q1A8_SCALES_BYTES + BONSAI_Q1A8_Q8_SUBBLOCKS * BONSAI_Q1A8_SUBBLOCK_STREAM_BYTES)
/* v4 weights stream and on-DDR packed layout: 144 B per Q1 block per rowblock. */
#define BONSAI_Q1A8_PACKED_PER_Q1_BLOCK \
    (BONSAI_Q1A8_SCALES_BYTES + BONSAI_Q1A8_Q8_SUBBLOCKS * BONSAI_Q1A8_WBITS_BYTES)
/* v4 acts stream (per Q1 block, broadcast across rowblocks): 160 B. */
#define BONSAI_Q1A8_ACTS_PER_Q1_BLOCK \
    (BONSAI_Q1A8_Q8_SUBBLOCKS * (BONSAI_Q1A8_ACTS_BYTES + BONSAI_Q1A8_ACT_SCALE_BYTES))

/* Compile-time assertions: packed weight bytes = Q1_0 source bytes (no
 * size change between layouts) and stream bytes match what the RTL emits. */
#if BONSAI_Q1A8_PACKED_PER_Q1_BLOCK != \
    (BONSAI_Q1A8_ROWS_PER_BLOCK * BONSAI_Q1_BLOCK_BYTES)
#error "packed Q1A8 weight layout size must equal Q1_0 source size"
#endif

/* Size helpers used by host and board to size buffers. */

static inline size_t bonsai_q1a8_packed_nbytes(uint32_t rows, uint32_t k) {
    /* round up rows to ROWS_PER_BLOCK so partial trailing rowblocks still fit */
    uint32_t rowblocks = (rows + BONSAI_Q1A8_ROWS_PER_BLOCK - 1)
                       / BONSAI_Q1A8_ROWS_PER_BLOCK;
    uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    return (size_t) rowblocks * blocks_per_row * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;
}

static inline size_t bonsai_q1a8_stream_nbytes(uint32_t rows, uint32_t k,
                                               uint32_t cols) {
    uint32_t rowblocks = (rows + BONSAI_Q1A8_ROWS_PER_BLOCK - 1)
                       / BONSAI_Q1A8_ROWS_PER_BLOCK;
    uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    return (size_t) cols * rowblocks * blocks_per_row
         * BONSAI_Q1A8_STREAM_PER_Q1_BLOCK;
}

static inline size_t bonsai_q1a8_stream_nbytes_per_col(uint32_t rows, uint32_t k) {
    return bonsai_q1a8_stream_nbytes(rows, k, 1);
}

/* Convenience: bytes that a single rowblock contributes to the stream over
 * all Q1 blocks in K. Useful when chunking by rowblock. */
static inline size_t bonsai_q1a8_stream_bytes_per_rowblock(uint32_t k) {
    uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    return (size_t) blocks_per_row * BONSAI_Q1A8_STREAM_PER_Q1_BLOCK;
}

static inline size_t bonsai_q1a8_packed_bytes_per_rowblock(uint32_t k) {
    uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    return (size_t) blocks_per_row * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;
}

/* v4 acts stream size (per matmul column). Sent once per kernel start. */
static inline size_t bonsai_q1a8_acts_stream_nbytes(uint32_t k) {
    uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    return (size_t) blocks_per_row * BONSAI_Q1A8_ACTS_PER_Q1_BLOCK;
}

/*
 * Repack Q1_0 weights into the packed AXIS layout.
 *
 *   q1_0_weights: row-major Q1_0, rows × (k / Q1_BLOCK) × Q1_BLOCK_BYTES bytes
 *   rows, k:      logical matmul dims; k must be a multiple of Q1_BLOCK
 *   out:          bonsai_q1a8_packed_nbytes(rows, k) bytes
 *
 * Trailing rows in a partial rowblock are zero-filled (yields zero
 * contributions). Returns 0 on success, negative on argument error.
 */
int bonsai_q1a8_pack_weights(
    const uint8_t * q1_0_weights,
    uint32_t rows,
    uint32_t k,
    uint8_t * out);

/*
 * Merge pre-packed weights + per-column quantized acts into the legacy v1
 * single-stream AXIS format. Kept only for the byte-identity parity test
 * against bonsai_pack_matmul_q1a8_stream. The v4 PL driver does not call
 * this — it DMAs packed_weights directly and packs acts separately.
 */
int bonsai_q1a8_merge_acts(
    const uint8_t * packed_weights,
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint32_t rows,
    uint32_t k,
    uint8_t * out_stream);

/*
 * Pack one column's acts + fp16 scales into the v4 acts wire stream.
 *
 *   act_quants:      k signed int8 (one column post-Q8_0 quantize)
 *   act_scale_bits:  k / Q8_BLOCK fp16 scales, little-endian u16
 *   k:               matmul inner dim, multiple of Q1_BLOCK
 *   out:             bonsai_q1a8_acts_stream_nbytes(k) bytes
 *
 * Output is rowblock-independent — same bytes feed every rowblock for this
 * column. The kernel loads it once into BRAM at start-of-column then
 * broadcasts during compute.
 */
int bonsai_q1a8_pack_acts(
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint32_t k,
    uint8_t * out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PYNQ_PROTO_Q1A8_LAYOUT_H */
