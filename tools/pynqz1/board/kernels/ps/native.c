#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    BONSAI_Q1_BLOCK = 128,
    BONSAI_Q1_BLOCK_BYTES = 18,
    BONSAI_Q8_BLOCK = 32,
    BONSAI_Q1A8_ROWS_PER_BLOCK = 8,
    BONSAI_Q1A8_SCALE_BEATS = (BONSAI_Q1A8_ROWS_PER_BLOCK + 3) / 4,
    BONSAI_Q1A8_WBITS_BEATS = (BONSAI_Q1A8_ROWS_PER_BLOCK + 1) / 2,
    BONSAI_Q1A8_ROWBLOCK_BYTES =
        (BONSAI_Q1A8_SCALE_BEATS + 4 * (5 + BONSAI_Q1A8_WBITS_BEATS)) * 8,
};

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

int bonsai_matmul_q1a8(
    const uint8_t * weights,
    const float * acts,
    float * dst,
    uint32_t rows,
    uint32_t cols,
    uint32_t k) {
    if (weights == NULL || acts == NULL || dst == NULL ||
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
    const uint32_t weight_row_bytes = blocks_per_row * BONSAI_Q1_BLOCK_BYTES;

    for (uint32_t col = 0; col < cols; ++col) {
        quantize_q8_0(acts + (size_t) col * k, k, act_quants, act_scales);

        for (uint32_t row = 0; row < rows; ++row) {
            float acc = 0.0f;
            const uint8_t * weight_row =
                weights + (size_t) row * weight_row_bytes;

            for (uint32_t q1_index = 0; q1_index < blocks_per_row; ++q1_index) {
                const uint8_t * block =
                    weight_row + (size_t) q1_index * BONSAI_Q1_BLOCK_BYTES;
                const float weight_scale = half_to_float(read_le_u16(block));
                if (weight_scale == 0.0f) {
                    continue;
                }

                const uint8_t * bits = block + 2;
                const uint32_t q1_base = q1_index * BONSAI_Q1_BLOCK;
                for (uint32_t q8_local = 0;
                     q8_local < BONSAI_Q1_BLOCK;
                     q8_local += BONSAI_Q8_BLOCK) {
                    const uint32_t q8_base = q1_base + q8_local;
                    const float act_scale = act_scales[q8_base / BONSAI_Q8_BLOCK];
                    if (act_scale == 0.0f) {
                        continue;
                    }

                    int sub_sum = 0;
                    for (uint32_t i = 0; i < BONSAI_Q8_BLOCK; ++i) {
                        const uint32_t bit_index = q8_local + i;
                        const uint8_t bit_byte = bits[bit_index >> 3];
                        const int act = (int) act_quants[q8_base + i];
                        if ((bit_byte & (uint8_t) (1u << (bit_index & 7u))) != 0) {
                            sub_sum += act;
                        } else {
                            sub_sum -= act;
                        }
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
    const size_t rowblock_bytes = BONSAI_Q1A8_ROWBLOCK_BYTES * (size_t) blocks_per_row;
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
