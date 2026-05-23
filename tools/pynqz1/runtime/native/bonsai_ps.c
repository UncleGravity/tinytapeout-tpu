#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
    BONSAI_Q1_BLOCK = 128,
    BONSAI_Q1_BLOCK_BYTES = 18,
    BONSAI_Q8_BLOCK = 32,
};

const char * bonsai_ps_version(void) {
    return "bonsai_ps_v1";
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
