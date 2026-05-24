#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pynq_e2e {

bool same_floats(const std::vector<float> & lhs, const std::vector<float> & rhs, float tolerance) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float allowed = std::max(tolerance, tolerance * std::fabs(rhs[i]));
        if (std::fabs(lhs[i] - rhs[i]) > allowed) {
            return false;
        }
    }
    return true;
}

float fp16_roundtrip(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

float silu(float value) {
    if (value >= 0.0f) {
        return value / (1.0f + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return value * exp_value / (1.0f + exp_value);
}

namespace {

float matmul_weight(int64_t row, int64_t index) {
    const uint32_t hash =
        static_cast<uint32_t>(row * 2654435761u) ^
        static_cast<uint32_t>(index * 374761393u);
    return (hash & 1) != 0 ? 1.0f : -1.0f;
}

float matmul_act(int64_t col, int64_t index) {
    const int value = static_cast<int>((index * 7 + col * 13) % 19) - 9;
    return static_cast<float>(value) * (col == 0 ? 0.25f : -0.5f);
}

float expected_matmul_cell(
    const std::vector<float> & weights,
    const std::vector<float> & acts,
    int64_t row,
    int64_t col) {
    constexpr int64_t q1_block = 128;
    constexpr int64_t q8_block = 32;
    float acc = 0.0f;
    for (int64_t q1_start = 0; q1_start < k_matmul_k; q1_start += q1_block) {
        float weight_abs_sum = 0.0f;
        for (int64_t index = q1_start; index < q1_start + q1_block; ++index) {
            weight_abs_sum += std::fabs(weights[static_cast<size_t>(row * k_matmul_k + index)]);
        }
        const float weight_scale = fp16_roundtrip(weight_abs_sum / q1_block);
        for (int64_t q8_start = q1_start; q8_start < q1_start + q1_block; q8_start += q8_block) {
            float amax = 0.0f;
            for (int64_t index = q8_start; index < q8_start + q8_block; ++index) {
                amax = std::max(amax, std::fabs(acts[static_cast<size_t>(col * k_matmul_k + index)]));
            }
            if (amax == 0.0f) continue;
            const float act_scale = fp16_roundtrip(amax / 127.0f);
            const float inv_scale = 127.0f / amax;
            int sub_sum = 0;
            for (int64_t index = q8_start; index < q8_start + q8_block; ++index) {
                long quant = std::lround(acts[static_cast<size_t>(col * k_matmul_k + index)] * inv_scale);
                quant = std::max(-128L, std::min(127L, quant));
                const float weight = weights[static_cast<size_t>(row * k_matmul_k + index)];
                sub_sum += weight >= 0.0f ? static_cast<int>(quant) : -static_cast<int>(quant);
            }
            acc += weight_scale * act_scale * static_cast<float>(sub_sum);
        }
    }
    return acc;
}

} // namespace

std::vector<float> make_matmul_weights() {
    std::vector<float> w(k_matmul_rows * k_matmul_k);
    for (int64_t row = 0; row < k_matmul_rows; ++row) {
        for (int64_t i = 0; i < k_matmul_k; ++i) {
            w[static_cast<size_t>(row * k_matmul_k + i)] = matmul_weight(row, i);
        }
    }
    return w;
}

std::vector<float> make_matmul_acts() {
    std::vector<float> a(k_matmul_cols * k_matmul_k);
    for (int64_t col = 0; col < k_matmul_cols; ++col) {
        for (int64_t i = 0; i < k_matmul_k; ++i) {
            a[static_cast<size_t>(col * k_matmul_k + i)] = matmul_act(col, i);
        }
    }
    return a;
}

std::vector<float> make_glue_input() {
    return { 0.5f, -1.0f, 2.0f, -4.0f, 1.5f, -2.0f, 3.0f, -6.0f };
}

std::vector<float> make_glue_bias() {
    return { 1.0f, 0.5f, -0.25f, 2.0f };
}

std::vector<float> make_swiglu_up() {
    return { 1.0f, 0.5f, -0.25f, 2.0f, 1.5f, -0.5f, 0.25f, -2.0f };
}

std::vector<float> expected_matmul(const std::vector<float> & weights, const std::vector<float> & acts) {
    std::vector<float> output(k_matmul_rows * k_matmul_cols);
    for (int64_t col = 0; col < k_matmul_cols; ++col) {
        for (int64_t row = 0; row < k_matmul_rows; ++row) {
            output[static_cast<size_t>(col * k_matmul_rows + row)] =
                expected_matmul_cell(weights, acts, row, col);
        }
    }
    return output;
}

std::vector<float> expected_glue_output(const std::vector<float> & input, const std::vector<float> & bias) {
    std::vector<float> scaled(input.size());
    for (int64_t col = 0; col < k_glue_cols; ++col) {
        for (int64_t row = 0; row < k_glue_rows; ++row) {
            const size_t i = static_cast<size_t>(col * k_glue_rows + row);
            scaled[i] = (input[i] + bias[static_cast<size_t>(row)]) * 0.5f;
        }
    }
    std::vector<float> output(input.size());
    for (int64_t col = 0; col < k_glue_cols; ++col) {
        float mean_square = 0.0f;
        for (int64_t row = 0; row < k_glue_rows; ++row) {
            const float v = scaled[static_cast<size_t>(col * k_glue_rows + row)];
            mean_square += v * v;
        }
        mean_square /= static_cast<float>(k_glue_rows);
        const float rms_scale = 1.0f / std::sqrt(mean_square + 1.0e-6f);
        for (int64_t row = 0; row < k_glue_rows; ++row) {
            const size_t i = static_cast<size_t>(col * k_glue_rows + row);
            output[i] = silu(scaled[i] * rms_scale) * bias[static_cast<size_t>(row)];
        }
    }
    return output;
}

std::vector<float> expected_swiglu_output(const std::vector<float> & gate, const std::vector<float> & up) {
    std::vector<float> output(gate.size());
    for (size_t i = 0; i < gate.size(); ++i) {
        output[i] = silu(gate[i]) * up[i];
    }
    return output;
}

bool quantize_matmul_weights(const std::vector<float> & weights, std::vector<uint8_t> * q1) {
    q1->resize(static_cast<size_t>(k_matmul_rows) * ggml_row_size(GGML_TYPE_Q1_0, k_matmul_k));
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q1_0, weights.data(), q1->data(), 0, k_matmul_rows, k_matmul_k, nullptr);
    return written == q1->size();
}

} // namespace pynq_e2e
