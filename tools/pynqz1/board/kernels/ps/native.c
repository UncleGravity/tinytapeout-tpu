#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "proto/q1a8_layout.h"

/* Block-size constants now live in proto/q1a8_layout.h. Old
 * BONSAI_Q1A8_ROWBLOCK_BYTES is BONSAI_Q1A8_STREAM_PER_Q1_BLOCK there. */

const char * bonsai_ps_version(void) {
    return "bonsai_ps_v1";
}

static float silu_f32(float value) {
    if (value >= 0.0f) {
        return value / (1.0f + expf(-value));
    }

    const float exp_value = expf(value);
    return value * exp_value / (1.0f + exp_value);
}

static uint16_t read_le_u16(const uint8_t * ptr) {
    return (uint16_t) ptr[0] | ((uint16_t) ptr[1] << 8);
}

static uint32_t read_le_u32(const uint8_t * ptr) {
    return (uint32_t) ptr[0]
         | ((uint32_t) ptr[1] << 8)
         | ((uint32_t) ptr[2] << 16)
         | ((uint32_t) ptr[3] << 24);
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t) h & 0x8000u) << 16;
    int32_t exp = (int32_t) ((h >> 10) & 0x1fu);
    uint32_t mant = (uint32_t) h & 0x03ffu;
    uint32_t bits = 0;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            ++exp;
            mant &= 0x03ffu;
            bits = sign | (uint32_t) (exp + 112) << 23 | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (uint32_t) (exp + 112) << 23 | (mant << 13);
    }

    float out = 0.0f;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));

    const uint16_t sign = (uint16_t) ((bits >> 16) & 0x8000u);
    int32_t exp = (int32_t) ((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }

        mant |= 0x00800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t rest = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit != 0 && (rest != 0 || (half_mant & 1u) != 0)) {
            ++half_mant;
        }
        return (uint16_t) (sign | half_mant);
    }

    if (exp >= 31) {
        if (mant == 0) {
            return (uint16_t) (sign | 0x7c00u);
        }
        return (uint16_t) (sign | 0x7c00u | (mant >> 13) | 1u);
    }

    uint32_t half_mant = mant >> 13;
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half_mant & 1u) != 0)) {
        ++half_mant;
        if (half_mant == 0x0400u) {
            half_mant = 0;
            ++exp;
            if (exp >= 31) {
                return (uint16_t) (sign | 0x7c00u);
            }
        }
    }

    return (uint16_t) (sign | ((uint16_t) exp << 10) | (uint16_t) half_mant);
}

static float fp16_roundtrip(float value) {
    return half_to_float(float_to_half(value));
}

static int lround_like_python(float value) {
    if (value >= 0.0f) {
        return (int) (value + 0.5f);
    }
    return (int) (value - 0.5f);
}

static void quantize_q8_0(
    const float * values,
    uint32_t k,
    int8_t * quants,
    float * scales) {
    for (uint32_t block_start = 0; block_start < k; block_start += BONSAI_Q8_BLOCK) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            const float value = values[block_start + i];
            if (isfinite(value)) {
                const float abs_value = value < 0.0f ? -value : value;
                if (abs_value > amax) {
                    amax = abs_value;
                }
            }
        }

        const uint32_t block_index = block_start / BONSAI_Q8_BLOCK;
        scales[block_index] = 0.0f;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            quants[block_start + i] = 0;
        }
        if (amax == 0.0f) {
            continue;
        }

        const float scale = amax / 127.0f;
        scales[block_index] = fp16_roundtrip(scale);
        const float inv_scale = 1.0f / scale;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            const float value = values[block_start + i];
            if (!isfinite(value)) {
                continue;
            }

            int quant = lround_like_python(value * inv_scale);
            if (quant > 127) {
                quant = 127;
            } else if (quant < -128) {
                quant = -128;
            }
            quants[block_start + i] = (int8_t) quant;
        }
    }
}

/*
 * Matmul Q1A8 over weights pre-packed in the AXIS rowblock layout
 * (proto/q1a8_layout.h). The host backend repacks Q1_0 weights on upload,
 * so this is what the daemon's PS path always sees. Used as the fallback
 * when no PL bitstream is loaded.
 *
 * Packed layout per (rowblock, q1_block) = 144 bytes:
 *   bytes [ 0..15]   weight_scales: 8 fp16 (row R at offset 2*R)
 *   bytes [16..47]   wbits sub-block 0: 8 u32 (row R at offset R*4)
 *   bytes [48..79]   wbits sub-block 1
 *   bytes [80..111]  wbits sub-block 2
 *   bytes [112..143] wbits sub-block 3
 */
int bonsai_matmul_q1a8(
    const uint8_t * packed_weights,
    const float * acts,
    float * dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t k) {
    if (packed_weights == NULL || acts == NULL || dst == NULL ||
        rows == 0 || cols == 0 || k == 0 ||
        (k % BONSAI_Q1_BLOCK) != 0) {
        return -1;
    }

    int8_t * act_quants = (int8_t *) malloc((size_t) k);
    float * act_scales = (float *) malloc((size_t) (k / BONSAI_Q8_BLOCK) * sizeof(float));
    if (act_quants == NULL || act_scales == NULL) {
        free(act_quants);
        free(act_scales);
        return -2;
    }

    const uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    const size_t   packed_per_rowblock =
        (size_t) blocks_per_row * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;

    for (uint32_t col = 0; col < cols; ++col) {
        quantize_q8_0(acts + (size_t) col * k, k, act_quants, act_scales);

        for (uint32_t row = 0; row < rows; ++row) {
            const uint32_t rb = row / BONSAI_Q1A8_ROWS_PER_BLOCK;
            const uint32_t row_in_rb = row % BONSAI_Q1A8_ROWS_PER_BLOCK;
            const uint8_t * rb_base =
                packed_weights + (size_t) rb * packed_per_rowblock;
            float acc = 0.0f;

            for (uint32_t q1 = 0; q1 < blocks_per_row; ++q1) {
                const uint8_t * block =
                    rb_base + (size_t) q1 * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;
                const float weight_scale = half_to_float(
                    read_le_u16(block + 2 * row_in_rb));
                if (weight_scale == 0.0f) continue;

                const uint8_t * wbits_base = block + BONSAI_Q1A8_SCALES_BYTES;
                for (uint32_t sub = 0; sub < BONSAI_Q1A8_Q8_SUBBLOCKS; ++sub) {
                    const uint32_t q8_base =
                        q1 * BONSAI_Q1_BLOCK + sub * BONSAI_Q8_BLOCK;
                    const float act_scale = act_scales[q8_base / BONSAI_Q8_BLOCK];
                    if (act_scale == 0.0f) continue;

                    /* This row's 32-bit weight mask for this sub-block. */
                    const uint32_t wbits = read_le_u32(
                        wbits_base
                        + (size_t) sub * BONSAI_Q1A8_WBITS_BYTES
                        + (size_t) row_in_rb * 4);

                    int sub_sum = 0;
                    for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
                        const int act = (int) act_quants[q8_base + i];
                        if (((wbits >> i) & 1u) != 0) sub_sum += act;
                        else                          sub_sum -= act;
                    }
                    acc += weight_scale * act_scale * (float) sub_sum;
                }
            }

            dst[(size_t) col * rows + row] = acc;
        }
    }

    free(act_quants);
    free(act_scales);
    return 0;
}

int bonsai_add_f32(
    const float * src0,
    const float * src1,
    float * dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t src1_broadcast) {
    if (src0 == NULL || src1 == NULL || dst == NULL || rows == 0 || cols == 0) {
        return -1;
    }

    for (uint32_t col = 0; col < cols; ++col) {
        const size_t col_offset = (size_t) col * rows;
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t index = col_offset + row;
            const size_t rhs_index = src1_broadcast != 0 ? row : index;
            dst[index] = src0[index] + src1[rhs_index];
        }
    }

    return 0;
}

int bonsai_mul_f32(
    const float * src0,
    const float * src1,
    float * dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t src1_broadcast) {
    if (src0 == NULL || src1 == NULL || dst == NULL || rows == 0 || cols == 0) {
        return -1;
    }

    for (uint32_t col = 0; col < cols; ++col) {
        const size_t col_offset = (size_t) col * rows;
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t index = col_offset + row;
            const size_t rhs_index = src1_broadcast != 0 ? row : index;
            dst[index] = src0[index] * src1[rhs_index];
        }
    }

    return 0;
}

int bonsai_scale_f32(
    const float * src,
    float * dst,
    uint32_t elements,
    float scale,
    float bias) {
    if (src == NULL || dst == NULL || elements == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < elements; ++i) {
        dst[i] = src[i] * scale + bias;
    }

    return 0;
}

int bonsai_silu_f32(
    const float * src,
    float * dst,
    uint32_t elements) {
    if (src == NULL || dst == NULL || elements == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < elements; ++i) {
        dst[i] = silu_f32(src[i]);
    }

    return 0;
}

int bonsai_swiglu_f32(
    const float * gate,
    const float * up,
    float * dst,
    uint32_t elements) {
    if (gate == NULL || up == NULL || dst == NULL || elements == 0) {
        return -1;
    }

    for (uint32_t i = 0; i < elements; ++i) {
        dst[i] = silu_f32(gate[i]) * up[i];
    }

    return 0;
}

/*
 * ROPE F32 — rotary position embedding for a [head_dim, n_head, n_token]
 * tensor in row-major (n_token slowest, head_dim fastest) layout.
 *
 * mode bit 1 (GGML_ROPE_TYPE_NEOX = 2) selects NEOX-style element pairing
 * (i pairs with i + n_dims/2). Otherwise NORMAL-style (i pairs with i+1).
 * Dimensions >= n_dims are copied through unmodified.
 *
 * Only implements the standard ROPE: no YaRN scaling (ext_factor/attn_factor/
 * beta_*) and no freq_factors tensor. The host lowering predicate refuses
 * to lower nodes with non-default YaRN params.
 */
/* YaRN per-dim correction bound (matches ggml's yarn_corr_dim):
 *   n_dims * log(n_ctx_orig / (n_rot * 2π)) / (2 * log(freq_base))
 *
 * Inline pi literal — M_PI isn't in C99 strict, and the board cross-compile
 * (cc -std=c99 without _GNU_SOURCE) doesn't pick it up from <math.h>.
 */
#define BONSAI_TWO_PI 6.283185307179586f

static float bonsai_yarn_corr_dim(int n_dims, int n_ctx_orig,
                                  float n_rot, float freq_base) {
    return (float) n_dims
        * logf((float) n_ctx_orig / (n_rot * BONSAI_TWO_PI))
        / (2.0f * logf(freq_base));
}

static float bonsai_yarn_ramp(float low, float high, int i0) {
    /* y = (i0/2 - low) / max(0.001, high - low); ramp = 1 - clamp01(y) */
    const float denom = (high - low) > 0.001f ? (high - low) : 0.001f;
    const float y = ((float) (i0 / 2) - low) / denom;
    const float cl = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    return 1.0f - cl;
}

int bonsai_rope_f32(
    const float * src,
    const int32_t * positions,
    float * dst,
    uint32_t head_dim,
    uint32_t n_head,
    uint32_t n_token,
    uint32_t n_dims,
    uint32_t mode,
    uint32_t n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow) {
    if (src == NULL || positions == NULL || dst == NULL) return -1;
    if (head_dim == 0 || n_head == 0 || n_token == 0) return -2;
    if (n_dims == 0 || n_dims > head_dim || (n_dims & 1)) return -3;

    const int is_neox = (mode & 2) != 0;
    const int use_yarn = (ext_factor != 0.0f);
    const float inv_n_dims = 1.0f / (float) n_dims;

    /* YaRN correction-dim window — constant across tokens/heads/dims. */
    float corr_low  = 0.0f;
    float corr_high = (float) n_dims;
    float mscale    = attn_factor;
    if (use_yarn) {
        const int ctx_orig = (n_ctx_orig > 0) ? (int) n_ctx_orig : 1;
        const float start = floorf(
            bonsai_yarn_corr_dim((int) n_dims, ctx_orig, beta_fast, freq_base));
        const float end   = ceilf(
            bonsai_yarn_corr_dim((int) n_dims, ctx_orig, beta_slow, freq_base));
        corr_low  = start < 0.0f ? 0.0f : start;
        corr_high = end > (float) (n_dims - 1) ? (float) (n_dims - 1) : end;
        /* YaRN mscale magnitude correction (matches ggml's rope_yarn). */
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    for (uint32_t t = 0; t < n_token; ++t) {
        const float pos = (float) positions[t];
        for (uint32_t h = 0; h < n_head; ++h) {
            const size_t off = ((size_t) t * n_head + h) * head_dim;
            const float * in  = src + off;
            float       * out = dst + off;

            for (uint32_t i = 0; i < n_dims; i += 2) {
                const float theta_extrap = pos
                    * powf(freq_base, -((float) i) * inv_n_dims);
                float theta;
                if (use_yarn) {
                    const float theta_interp = freq_scale * theta_extrap;
                    const float ramp_mix = bonsai_yarn_ramp(corr_low, corr_high,
                                                            (int) i) * ext_factor;
                    theta = theta_interp * (1.0f - ramp_mix)
                          + theta_extrap * ramp_mix;
                } else {
                    theta = theta_extrap * freq_scale;
                }
                const float c = cosf(theta) * mscale;
                const float s = sinf(theta) * mscale;

                uint32_t i0, i1;
                if (is_neox) {
                    i0 = i / 2;
                    i1 = i / 2 + n_dims / 2;
                } else {
                    i0 = i;
                    i1 = i + 1;
                }

                const float x0 = in[i0];
                const float x1 = in[i1];
                out[i0] = x0 * c - x1 * s;
                out[i1] = x0 * s + x1 * c;
            }
            for (uint32_t i = n_dims; i < head_dim; ++i) {
                out[i] = in[i];
            }
        }
    }
    return 0;
}

int bonsai_rms_norm_f32(
    const float * src,
    float * dst,
    uint32_t rows,
    uint32_t cols,
    float eps) {
    if (src == NULL || dst == NULL || rows == 0 || cols == 0) {
        return -1;
    }

    for (uint32_t col = 0; col < cols; ++col) {
        const size_t col_offset = (size_t) col * rows;
        float mean_square = 0.0f;
        for (uint32_t row = 0; row < rows; ++row) {
            const float value = src[col_offset + row];
            mean_square += value * value;
        }

        const float scale = 1.0f / sqrtf(mean_square / (float) rows + eps);
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t index = col_offset + row;
            dst[index] = src[index] * scale;
        }
    }

    return 0;
}

// Q8_0 quantize that emits int8 quants + fp16 scale bits (the form the PL
// rowblock kernel wants on the wire). Matches quantize_q8_0() above bit-for-bit
// modulo the scale representation: a float roundtrip through fp16 there, the
// raw fp16 bits here.
int bonsai_quantize_q8_0_pl(
    const float * values,
    uint32_t k,
    int8_t * out_quants,
    uint16_t * out_scale_bits) {
    if (values == NULL || out_quants == NULL || out_scale_bits == NULL) {
        return -1;
    }
    if (k == 0 || (k % BONSAI_Q8_BLOCK) != 0) {
        return -2;
    }

    for (uint32_t block_start = 0; block_start < k; block_start += BONSAI_Q8_BLOCK) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            const float value = values[block_start + i];
            if (isfinite(value)) {
                const float abs_value = value < 0.0f ? -value : value;
                if (abs_value > amax) {
                    amax = abs_value;
                }
            }
        }

        const uint32_t block_index = block_start / BONSAI_Q8_BLOCK;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            out_quants[block_start + i] = 0;
        }
        if (amax == 0.0f) {
            out_scale_bits[block_index] = 0;
            continue;
        }

        const float scale = amax / 127.0f;
        out_scale_bits[block_index] = float_to_half(scale);
        const float inv_scale = 1.0f / scale;
        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
            const float value = values[block_start + i];
            if (!isfinite(value)) {
                continue;
            }

            int quant = lround_like_python(value * inv_scale);
            if (quant > 127) {
                quant = 127;
            } else if (quant < -128) {
                quant = -128;
            }
            out_quants[block_start + i] = (int8_t) quant;
        }
    }

    return 0;
}

// Write a little-endian uint64 at `out`, advancing `*cursor`.
static void write_le_u64(uint8_t * out, size_t * cursor, uint64_t value) {
    uint8_t * p = out + *cursor;
    p[0] = (uint8_t) (value);
    p[1] = (uint8_t) (value >> 8);
    p[2] = (uint8_t) (value >> 16);
    p[3] = (uint8_t) (value >> 24);
    p[4] = (uint8_t) (value >> 32);
    p[5] = (uint8_t) (value >> 40);
    p[6] = (uint8_t) (value >> 48);
    p[7] = (uint8_t) (value >> 56);
    *cursor += 8;
}

// Pack one rowblock (up to ROWS_PER_BLOCK rows starting at row_start) into the
// AXIS wire stream. Lanes beyond row_count are zero-padded so they contribute
// zero in the kernel's signed reducer.
static void pack_rowblock(
    const uint8_t * weights,
    uint32_t row_start,
    uint32_t row_count,
    uint32_t weight_row_bytes,
    uint32_t blocks_per_row,
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint8_t * out) {
    size_t cursor = 0;

    for (uint32_t q1_index = 0; q1_index < blocks_per_row; ++q1_index) {
        // Weight scales: SCALE_BEATS beats of four packed fp16 scales each.
        for (uint32_t beat = 0; beat < BONSAI_Q1A8_SCALE_BEATS; ++beat) {
            uint64_t word = 0;
            for (uint32_t local = 0; local < 4; ++local) {
                const uint32_t lane = beat * 4 + local;
                uint16_t scale = 0;
                if (lane < row_count) {
                    const size_t off =
                        (size_t) (row_start + lane) * weight_row_bytes
                        + (size_t) q1_index * BONSAI_Q1_BLOCK_BYTES;
                    scale = (uint16_t) weights[off]
                          | ((uint16_t) weights[off + 1] << 8);
                }
                word |= ((uint64_t) scale) << (local * 16);
            }
            write_le_u64(out, &cursor, word);
        }

        // For each Q8 sub-block (0, 32, 64, 96).
        for (uint32_t q8_local = 0; q8_local < BONSAI_Q1_BLOCK; q8_local += BONSAI_Q8_BLOCK) {
            const uint32_t q8_base = q1_index * BONSAI_Q1_BLOCK + q8_local;

            // 32 int8 acts: 4 u64 beats, little-endian.
            for (uint32_t beat = 0; beat < 4; ++beat) {
                uint64_t word = 0;
                for (uint32_t byte = 0; byte < 8; ++byte) {
                    const uint8_t v = (uint8_t) act_quants[q8_base + beat * 8 + byte];
                    word |= ((uint64_t) v) << (byte * 8);
                }
                write_le_u64(out, &cursor, word);
            }

            // Activation scale (fp16 bits in low 16, rest zero).
            write_le_u64(out, &cursor, (uint64_t) act_scale_bits[q8_base / BONSAI_Q8_BLOCK]);

            // Weight bits: WBITS_BEATS beats of two packed u32 each.
            for (uint32_t beat = 0; beat < BONSAI_Q1A8_WBITS_BEATS; ++beat) {
                uint64_t word = 0;
                for (uint32_t local = 0; local < 2; ++local) {
                    const uint32_t lane = beat * 2 + local;
                    uint32_t bits = 0;
                    if (lane < row_count) {
                        const size_t base =
                            (size_t) (row_start + lane) * weight_row_bytes
                            + (size_t) q1_index * BONSAI_Q1_BLOCK_BYTES
                            + 2  // skip fp16 scale
                            + q8_local / 8;
                        bits = (uint32_t) weights[base]
                             | ((uint32_t) weights[base + 1] << 8)
                             | ((uint32_t) weights[base + 2] << 16)
                             | ((uint32_t) weights[base + 3] << 24);
                    }
                    word |= ((uint64_t) bits) << (local * 32);
                }
                write_le_u64(out, &cursor, word);
            }
        }
    }
}

/*
 * GET_ROWS: dst[i] = src0[indices[i]], dequantized to F32.
 *
 * Three entry points, one per src0 type. dst is always F32, indices are
 * always I32 (ggml uses I32 for GET_ROWS; SET_ROWS is the one with both
 * I32/I64). Strides are in BYTES.
 *
 * Outer loops match ggml-cpu/ops.cpp:4663: for (i12, i11, i10), look up
 * i01 = indices[i10*nb10 + i11*nb11 + i12*nb12], then copy src0 row
 * [i01, i11, i12] to dst row [i10, i11, i12]. nb10 is implicit
 * (= sizeof(int32_t)).
 */
static const int32_t * indices_i32_at(
    const uint8_t * indices, size_t i10,
    size_t i11, size_t indices_nb1,
    size_t i12, size_t indices_nb2) {
    return (const int32_t *)(indices
        + i10 * sizeof(int32_t)
        + i11 * indices_nb1
        + i12 * indices_nb2);
}

int bonsai_get_rows_f32(
    const uint8_t * src0,
    size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
    const uint8_t * indices,
    size_t indices_nb1, size_t indices_nb2,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
    uint32_t head_dim, uint32_t ne01,
    uint32_t ne10, uint32_t ne11, uint32_t ne12) {
    if (src0 == NULL || indices == NULL || dst == NULL || head_dim == 0) {
        return -1;
    }
    for (uint32_t i12 = 0; i12 < ne12; ++i12) {
        for (uint32_t i11 = 0; i11 < ne11; ++i11) {
            for (uint32_t i10 = 0; i10 < ne10; ++i10) {
                const int32_t i01 = *indices_i32_at(
                    indices, i10, i11, indices_nb1, i12, indices_nb2);
                if (i01 < 0 || (uint32_t) i01 >= ne01) return -2;

                const float * src_row = (const float *)(src0
                    + (size_t) i01 * src0_nb1
                    + (size_t) i11 * src0_nb2
                    + (size_t) i12 * src0_nb3);
                float * dst_row = (float *)(dst
                    + (size_t) i10 * dst_nb1
                    + (size_t) i11 * dst_nb2
                    + (size_t) i12 * dst_nb3);
                for (uint32_t d = 0; d < head_dim; ++d) dst_row[d] = src_row[d];
            }
        }
    }
    return 0;
}

int bonsai_get_rows_f16(
    const uint8_t * src0,
    size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
    const uint8_t * indices,
    size_t indices_nb1, size_t indices_nb2,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
    uint32_t head_dim, uint32_t ne01,
    uint32_t ne10, uint32_t ne11, uint32_t ne12) {
    if (src0 == NULL || indices == NULL || dst == NULL || head_dim == 0) {
        return -1;
    }
    for (uint32_t i12 = 0; i12 < ne12; ++i12) {
        for (uint32_t i11 = 0; i11 < ne11; ++i11) {
            for (uint32_t i10 = 0; i10 < ne10; ++i10) {
                const int32_t i01 = *indices_i32_at(
                    indices, i10, i11, indices_nb1, i12, indices_nb2);
                if (i01 < 0 || (uint32_t) i01 >= ne01) return -2;

                const uint16_t * src_row = (const uint16_t *)(src0
                    + (size_t) i01 * src0_nb1
                    + (size_t) i11 * src0_nb2
                    + (size_t) i12 * src0_nb3);
                float * dst_row = (float *)(dst
                    + (size_t) i10 * dst_nb1
                    + (size_t) i11 * dst_nb2
                    + (size_t) i12 * dst_nb3);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dst_row[d] = half_to_float(src_row[d]);
                }
            }
        }
    }
    return 0;
}

int bonsai_get_rows_q1_0(
    const uint8_t * src0,
    size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
    const uint8_t * indices,
    size_t indices_nb1, size_t indices_nb2,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
    uint32_t head_dim, uint32_t ne01,
    uint32_t ne10, uint32_t ne11, uint32_t ne12) {
    if (src0 == NULL || indices == NULL || dst == NULL ||
        head_dim == 0 || (head_dim % BONSAI_Q1_BLOCK) != 0) {
        return -1;
    }
    (void) src0_nb1;
    (void) src0_nb2;
    (void) src0_nb3;

    const uint32_t blocks_per_row = head_dim / BONSAI_Q1_BLOCK;
    const size_t packed_rowblock_bytes =
        (size_t) blocks_per_row * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;

    for (uint32_t i12 = 0; i12 < ne12; ++i12) {
        for (uint32_t i11 = 0; i11 < ne11; ++i11) {
            for (uint32_t i10 = 0; i10 < ne10; ++i10) {
                const int32_t i01 = *indices_i32_at(
                    indices, i10, i11, indices_nb1, i12, indices_nb2);
                if (i01 < 0 || (uint32_t) i01 >= ne01) return -2;

                float * dst_row = (float *)(dst
                    + (size_t) i10 * dst_nb1
                    + (size_t) i11 * dst_nb2
                    + (size_t) i12 * dst_nb3);

                const uint32_t rb = (uint32_t) i01 / BONSAI_Q1A8_ROWS_PER_BLOCK;
                const uint32_t lane = (uint32_t) i01 % BONSAI_Q1A8_ROWS_PER_BLOCK;
                const uint8_t * rb_base =
                    src0 + (size_t) rb * packed_rowblock_bytes;

                for (uint32_t b = 0; b < blocks_per_row; ++b) {
                    const uint8_t * blk =
                        rb_base + (size_t) b * BONSAI_Q1A8_PACKED_PER_Q1_BLOCK;
                    const uint8_t * scale_ptr =
                        blk + (size_t) (lane / 4) * 8 + (size_t) (lane % 4) * 2;
                    const float scale = half_to_float(read_le_u16(scale_ptr));
                    float * out = dst_row + (size_t) b * BONSAI_Q1_BLOCK;

                    for (uint32_t sub = 0; sub < BONSAI_Q1A8_Q8_SUBBLOCKS; ++sub) {
                        const uint8_t * sub_ptr =
                            blk + BONSAI_Q1A8_SCALES_BYTES
                            + (size_t) sub * BONSAI_Q1A8_WBITS_BYTES
                            + (size_t) (lane / 2) * 8
                            + (size_t) (lane % 2) * 4;
                        const uint32_t bits = read_le_u32(sub_ptr);
                        for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
                            const uint32_t out_i = sub * BONSAI_Q8_BLOCK + i;
                            const int bit = (bits >> i) & 1;
                            out[out_i] = bit ? scale : -scale;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

/*
 * SET_ROWS: writes src0 (F32) rows into dst (F16) at indices in src1.
 * Mirrors ggml-cpu/ops.cpp:4904 with the F32→F16 from_float specialization.
 *
 * Two entry points by indices type (I32 vs I64). All other strides in BYTES.
 *   for i03 in 0..ne03:
 *     for i02 in 0..ne02:
 *       for i in 0..ne01:
 *         i12 = i03 % ne12
 *         i11 = i02 % ne11
 *         i10 = i
 *         row = indices[i10*nb10 + i11*nb11 + i12*nb12]
 *         dst[row, i02, i03] := f16( src0[i, i02, i03] )
 */
int bonsai_set_rows_f32_to_f16_i32(
    const uint8_t * src0,
    size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
    const uint8_t * indices,
    size_t indices_nb1, size_t indices_nb2,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
    uint32_t head_dim,
    uint32_t ne01, uint32_t ne02, uint32_t ne03,
    uint32_t ne11, uint32_t ne12) {
    if (src0 == NULL || indices == NULL || dst == NULL || head_dim == 0) {
        return -1;
    }
    for (uint32_t i03 = 0; i03 < ne03; ++i03) {
        for (uint32_t i02 = 0; i02 < ne02; ++i02) {
            const uint32_t i12 = i03 % ne12;
            const uint32_t i11 = i02 % ne11;
            for (uint32_t i = 0; i < ne01; ++i) {
                const int32_t row = *indices_i32_at(
                    indices, i, i11, indices_nb1, i12, indices_nb2);
                if (row < 0) return -2;

                const float * src_row = (const float *)(src0
                    + (size_t) i   * src0_nb1
                    + (size_t) i02 * src0_nb2
                    + (size_t) i03 * src0_nb3);
                uint16_t * dst_row = (uint16_t *)(dst
                    + (size_t) row * dst_nb1
                    + (size_t) i02 * dst_nb2
                    + (size_t) i03 * dst_nb3);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dst_row[d] = float_to_half(src_row[d]);
                }
            }
        }
    }
    return 0;
}

int bonsai_set_rows_f32_to_f16_i64(
    const uint8_t * src0,
    size_t src0_nb1, size_t src0_nb2, size_t src0_nb3,
    const uint8_t * indices,
    size_t indices_nb1, size_t indices_nb2,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2, size_t dst_nb3,
    uint32_t head_dim,
    uint32_t ne01, uint32_t ne02, uint32_t ne03,
    uint32_t ne11, uint32_t ne12) {
    if (src0 == NULL || indices == NULL || dst == NULL || head_dim == 0) {
        return -1;
    }
    for (uint32_t i03 = 0; i03 < ne03; ++i03) {
        for (uint32_t i02 = 0; i02 < ne02; ++i02) {
            const uint32_t i12 = i03 % ne12;
            const uint32_t i11 = i02 % ne11;
            for (uint32_t i = 0; i < ne01; ++i) {
                const int64_t * idx_ptr = (const int64_t *)(indices
                    + (size_t) i   * sizeof(int64_t)
                    + (size_t) i11 * indices_nb1
                    + (size_t) i12 * indices_nb2);
                const int64_t row = *idx_ptr;
                if (row < 0) return -2;

                const float * src_row = (const float *)(src0
                    + (size_t) i   * src0_nb1
                    + (size_t) i02 * src0_nb2
                    + (size_t) i03 * src0_nb3);
                uint16_t * dst_row = (uint16_t *)(dst
                    + (size_t) row * dst_nb1
                    + (size_t) i02 * dst_nb2
                    + (size_t) i03 * dst_nb3);
                for (uint32_t d = 0; d < head_dim; ++d) {
                    dst_row[d] = float_to_half(src_row[d]);
                }
            }
        }
    }
    return 0;
}

/*
 * SET_ROWS fast path: convert one contiguous row of `head_dim` F32 values to
 * F16 and write it at `dst`. The Python kernel resolves each destination
 * row's slab address and calls this per row, so it never stages the whole
 * KV-cache tensor through scratch. Numerics match the bulk
 * bonsai_set_rows_f32_to_f16_* kernels above (same float_to_half).
 */
int bonsai_set_rows_row_f32_to_f16(
    uint8_t * dst, const uint8_t * src, uint32_t head_dim) {
    if (dst == NULL || src == NULL || head_dim == 0) {
        return -1;
    }
    const float * src_row = (const float *) src;
    uint16_t * dst_row = (uint16_t *) dst;
    for (uint32_t d = 0; d < head_dim; ++d) {
        dst_row[d] = float_to_half(src_row[d]);
    }
    return 0;
}

/*
 * FLASH_ATTN_EXT F32 — online softmax flash attention for Bonsai shape.
 *
 * Mirrors ggml-cpu/ops.cpp:8168 (ggml_compute_forward_flash_attn_ext_f16_one_chunk)
 * with these specializations: Q=F32, K=F16, V=F16, mask=F16 (or none),
 * GQA (n_head % n_head_kv == 0), no ALiBi, no softcap, no sinks.
 *
 * Tensor layout (ggml convention):
 *   q  : [head_dim_q, n_token, n_head,    1]   nb=[4, q_nb1, q_nb2, *]
 *   k  : [head_dim_q, n_kv,    n_head_kv, 1]   nb=[2, k_nb1, k_nb2, *]
 *   v  : [head_dim_v, n_kv,    n_head_kv, 1]   nb=[2, v_nb1, v_nb2, *]
 *   mask:[n_kv,       n_pad,   1,         1]   nb=[2, mask_nb1, *, *]   (optional)
 *   dst: [head_dim_v, n_head,  n_token,   1]   nb=[4, dst_nb1, dst_nb2, *]
 *
 * Note dst's middle dim is n_head and the slowest is n_token — this is
 * ggml's permute-on-write. dst_nb1 = head_dim_v*sizeof(float).
 *
 * Stack scratch: 4 * BONSAI_FATTN_MAX_HEAD_DIM floats = 4 KiB at HD=256.
 */
#define BONSAI_FATTN_MAX_HEAD_DIM 256

int bonsai_flash_attn_ext_f32(
    const uint8_t * q_data,
    size_t q_nb1, size_t q_nb2,
    const uint8_t * k_data,
    size_t k_nb1, size_t k_nb2,
    const uint8_t * v_data,
    size_t v_nb1, size_t v_nb2,
    const uint8_t * mask_data,   /* NULL if no mask */
    size_t mask_nb1,
    uint8_t * dst,
    size_t dst_nb1, size_t dst_nb2,
    uint32_t head_dim_q, uint32_t head_dim_v,
    uint32_t n_head, uint32_t n_head_kv,
    uint32_t n_kv, uint32_t n_token,
    float scale) {

    if (q_data == NULL || k_data == NULL || v_data == NULL || dst == NULL) {
        return -1;
    }
    if (head_dim_q == 0 || head_dim_v == 0 || n_head == 0 ||
        n_head_kv == 0 || n_kv == 0 || n_token == 0) {
        return -2;
    }
    if (head_dim_q > BONSAI_FATTN_MAX_HEAD_DIM ||
        head_dim_v > BONSAI_FATTN_MAX_HEAD_DIM) {
        return -3;
    }
    if ((n_head % n_head_kv) != 0) return -4;

    const uint32_t rk2 = n_head / n_head_kv;

    float Q_row[BONSAI_FATTN_MAX_HEAD_DIM];
    float VKQ[BONSAI_FATTN_MAX_HEAD_DIM];

    for (uint32_t iq1 = 0; iq1 < n_token; ++iq1) {
        const uint8_t * mask_row = mask_data
            ? (mask_data + (size_t) iq1 * mask_nb1)
            : NULL;

        for (uint32_t iq2 = 0; iq2 < n_head; ++iq2) {
            const uint32_t ik2 = iq2 / rk2;

            const float * q_ptr = (const float *)(q_data
                + (size_t) iq1 * q_nb1
                + (size_t) iq2 * q_nb2);
            for (uint32_t d = 0; d < head_dim_q; ++d) Q_row[d] = q_ptr[d];

            float M = -INFINITY;
            float S = 0.0f;
            for (uint32_t d = 0; d < head_dim_v; ++d) VKQ[d] = 0.0f;

            for (uint32_t ic = 0; ic < n_kv; ++ic) {
                float mv = 0.0f;
                if (mask_row != NULL) {
                    const uint16_t * mp = (const uint16_t *) mask_row;
                    mv = half_to_float(mp[ic]);
                    if (!isfinite(mv) && mv < 0.0f) continue;  /* -INF mask = skip */
                }

                /* Dot Q (F32) · K[ic] (F16). */
                const uint16_t * k_ptr = (const uint16_t *)(k_data
                    + (size_t) ic  * k_nb1
                    + (size_t) ik2 * k_nb2);
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim_q; ++d) {
                    s += Q_row[d] * half_to_float(k_ptr[d]);
                }
                s = s * scale + mv;

                /* Online softmax update — branchless form taken from
                 * the ggml reference. */
                const float Mold = M;
                float ms = 1.0f;
                float vs = 1.0f;
                if (s > M) {
                    M = s;
                    ms = expf(Mold - M);
                    for (uint32_t d = 0; d < head_dim_v; ++d) VKQ[d] *= ms;
                } else {
                    vs = expf(s - M);
                }

                /* VKQ += vs * V[ic] (F16 → F32 mad). */
                const uint16_t * v_ptr = (const uint16_t *)(v_data
                    + (size_t) ic  * v_nb1
                    + (size_t) ik2 * v_nb2);
                for (uint32_t d = 0; d < head_dim_v; ++d) {
                    VKQ[d] += vs * half_to_float(v_ptr[d]);
                }

                S = S * ms + vs;
            }

            /* Normalize and write to dst[head_dim_v, iq2, iq1]. */
            const float S_inv = (S == 0.0f) ? 0.0f : (1.0f / S);
            float * dst_ptr = (float *)(dst
                + (size_t) iq2 * dst_nb1
                + (size_t) iq1 * dst_nb2);
            for (uint32_t d = 0; d < head_dim_v; ++d) {
                dst_ptr[d] = VKQ[d] * S_inv;
            }
        }
    }
    return 0;
}

// Pack a full MATMUL_Q1A8 into the AXIS wire stream. Output layout:
//   out_stream[col][rowblock] = rowblock_bytes consecutive bytes
// rowblock count = ceil(rows / ROWS_PER_BLOCK). The last rowblock may be
// partial; lanes past the active row_count are zero-padded.
int bonsai_pack_matmul_q1a8_stream(
    const uint8_t * weights,
    const int8_t * act_quants,
    const uint16_t * act_scale_bits,
    uint32_t rows,
    uint32_t cols,
    uint32_t k,
    uint8_t * out_stream) {
    if (weights == NULL || act_quants == NULL || act_scale_bits == NULL ||
        out_stream == NULL || rows == 0 || cols == 0) {
        return -1;
    }
    if (k == 0 || (k % BONSAI_Q1_BLOCK) != 0) {
        return -2;
    }

    const uint32_t blocks_per_row = k / BONSAI_Q1_BLOCK;
    const uint32_t weight_row_bytes = blocks_per_row * BONSAI_Q1_BLOCK_BYTES;
    const uint32_t scales_per_col = k / BONSAI_Q8_BLOCK;
    const uint32_t rowblocks_per_col =
        (rows + BONSAI_Q1A8_ROWS_PER_BLOCK - 1) / BONSAI_Q1A8_ROWS_PER_BLOCK;
    const size_t rowblock_bytes = BONSAI_Q1A8_STREAM_PER_Q1_BLOCK * (size_t) blocks_per_row;
    const size_t col_stride = rowblock_bytes * rowblocks_per_col;

    for (uint32_t col = 0; col < cols; ++col) {
        const int8_t * col_quants = act_quants + (size_t) col * k;
        const uint16_t * col_scales = act_scale_bits + (size_t) col * scales_per_col;
        uint8_t * col_out = out_stream + (size_t) col * col_stride;

        for (uint32_t rb = 0; rb < rowblocks_per_col; ++rb) {
            const uint32_t row_start = rb * BONSAI_Q1A8_ROWS_PER_BLOCK;
            uint32_t row_count = rows - row_start;
            if (row_count > BONSAI_Q1A8_ROWS_PER_BLOCK) {
                row_count = BONSAI_Q1A8_ROWS_PER_BLOCK;
            }

            pack_rowblock(
                weights,
                row_start,
                row_count,
                weight_row_bytes,
                blocks_per_row,
                col_quants,
                col_scales,
                col_out + (size_t) rb * rowblock_bytes);
        }
    }

    return 0;
}
