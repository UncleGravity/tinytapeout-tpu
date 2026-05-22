#include "ggml-pynq.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int64_t k_matmul_k = 128;
constexpr int64_t k_matmul_rows = 3;
constexpr int64_t k_matmul_cols = 2;

float fp16_roundtrip(float value) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

bool same_floats(
    const std::vector<float> & lhs,
    const std::vector<float> & rhs,
    float tolerance = 1e-6f) {
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

std::vector<float> make_matmul_weights() {
    std::vector<float> weights(k_matmul_rows * k_matmul_k);
    for (int64_t row = 0; row < k_matmul_rows; ++row) {
        for (int64_t index = 0; index < k_matmul_k; ++index) {
            weights[static_cast<size_t>(row * k_matmul_k + index)] =
                matmul_weight(row, index);
        }
    }
    return weights;
}

std::vector<float> make_matmul_acts() {
    std::vector<float> acts(k_matmul_cols * k_matmul_k);
    for (int64_t col = 0; col < k_matmul_cols; ++col) {
        for (int64_t index = 0; index < k_matmul_k; ++index) {
            acts[static_cast<size_t>(col * k_matmul_k + index)] =
                matmul_act(col, index);
        }
    }
    return acts;
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
            weight_abs_sum += std::fabs(
                weights[static_cast<size_t>(row * k_matmul_k + index)]);
        }
        const float weight_scale = fp16_roundtrip(weight_abs_sum / q1_block);

        for (int64_t q8_start = q1_start; q8_start < q1_start + q1_block;
             q8_start += q8_block) {
            float amax = 0.0f;
            for (int64_t index = q8_start; index < q8_start + q8_block; ++index) {
                amax = std::max(
                    amax,
                    std::fabs(acts[static_cast<size_t>(col * k_matmul_k + index)]));
            }
            if (amax == 0.0f) {
                continue;
            }

            const float act_scale = fp16_roundtrip(amax / 127.0f);
            const float inv_scale = 127.0f / amax;
            int sub_sum = 0;
            for (int64_t index = q8_start; index < q8_start + q8_block; ++index) {
                long quant = std::lround(
                    acts[static_cast<size_t>(col * k_matmul_k + index)] * inv_scale);
                quant = std::max(-128L, std::min(127L, quant));
                const float weight =
                    weights[static_cast<size_t>(row * k_matmul_k + index)];
                sub_sum += weight >= 0.0f ? static_cast<int>(quant) :
                    -static_cast<int>(quant);
            }
            acc += weight_scale * act_scale * static_cast<float>(sub_sum);
        }
    }

    return acc;
}

std::vector<float> expected_matmul(
    const std::vector<float> & weights,
    const std::vector<float> & acts) {
    std::vector<float> output(k_matmul_rows * k_matmul_cols);
    for (int64_t col = 0; col < k_matmul_cols; ++col) {
        for (int64_t row = 0; row < k_matmul_rows; ++row) {
            output[static_cast<size_t>(col * k_matmul_rows + row)] =
                expected_matmul_cell(weights, acts, row, col);
        }
    }
    return output;
}

bool quantize_matmul_weights(
    const std::vector<float> & weights,
    std::vector<uint8_t> * q1_weights) {
    q1_weights->resize(
        static_cast<size_t>(k_matmul_rows) *
        ggml_row_size(GGML_TYPE_Q1_0, k_matmul_k));
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q1_0,
        weights.data(),
        q1_weights->data(),
        0,
        k_matmul_rows,
        k_matmul_k,
        nullptr);
    return written == q1_weights->size();
}

bool scheduler_matmul_smoke(
    ggml_backend_t backend,
    ggml_backend_t cpu_backend,
    const std::vector<uint8_t> & q1_weights,
    const std::vector<float> & acts,
    const std::vector<float> & expected,
    int * split_count) {
    constexpr size_t graph_size = 4;
    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) +
            4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "scheduler ggml_init failed\n");
        return false;
    }

    ggml_tensor * weights =
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, k_matmul_k, k_matmul_rows);
    ggml_tensor * input =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_matmul_k, k_matmul_cols);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        nullptr,
        2,
        graph_size,
        false,
        true);
    if (sched == nullptr || !ggml_backend_sched_alloc_graph(sched, graph)) {
        std::fprintf(stderr, "pynq scheduler graph allocation failed\n");
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched, output) != backend) {
        std::fprintf(stderr, "pynq scheduler did not assign MUL_MAT to PYNQ\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, q1_weights.data(), 0, q1_weights.size());
    ggml_backend_tensor_set(input, acts.data(), 0, acts.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "pynq scheduler MATMUL_Q1A8 compute failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    *split_count = ggml_backend_sched_get_n_splits(sched);
    const bool ok = same_floats(actual, expected, 1e-3f);
    if (!ok) {
        std::fprintf(stderr, "scheduler MATMUL_Q1A8 output mismatch\n");
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main() {
    ggml_backend_reg_t reg = ggml_backend_pynq_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (dev == nullptr) {
        std::fprintf(stderr, "pynq backend did not register a device\n");
        return 1;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    if (total_bytes == 0 || free_bytes == 0) {
        std::fprintf(stderr, "pynq backend could not read bonsaid memory\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "pynq backend failed HELLO\n");
        return 1;
    }

    std::vector<float> matmul_weights = make_matmul_weights();
    std::vector<float> matmul_acts = make_matmul_acts();
    std::vector<float> matmul_expected = expected_matmul(matmul_weights, matmul_acts);
    std::vector<uint8_t> matmul_q1_weights;
    if (!quantize_matmul_weights(matmul_weights, &matmul_q1_weights)) {
        std::fprintf(stderr, "Q1_0 weight quantization failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    constexpr size_t graph_size = 16;
    const ggml_init_params params = {
        /* .mem_size   = */ 12 * ggml_tensor_overhead() +
            2 * ggml_graph_overhead_custom(graph_size, false) +
            8192,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    constexpr int64_t n_values = 64;
    constexpr int64_t view_values = 8;
    constexpr int64_t view_start = 11;
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(tensor, "pynq-smoke-root");
    ggml_tensor * view = ggml_view_1d(
        ctx,
        tensor,
        view_values,
        view_start * sizeof(float));
    ggml_set_name(view, "pynq-smoke-view");
    ggml_tensor * copy_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(copy_dst, "pynq-smoke-copy-dst");
    ggml_tensor * copy_dst_view = ggml_view_1d(
        ctx,
        copy_dst,
        view_values,
        view_start * sizeof(float));
    ggml_set_name(copy_dst_view, "pynq-smoke-copy-dst-view");
    ggml_tensor * copy = ggml_cpy(ctx, view, copy_dst_view);
    ggml_set_name(copy, "pynq-smoke-copy");
    ggml_tensor * matmul_src0 =
        ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, k_matmul_k, k_matmul_rows);
    ggml_set_name(matmul_src0, "pynq-smoke-matmul-weights");
    ggml_tensor * matmul_src1 =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_matmul_k, k_matmul_cols);
    ggml_set_name(matmul_src1, "pynq-smoke-matmul-acts");
    ggml_tensor * matmul = ggml_mul_mat(ctx, matmul_src0, matmul_src1);
    ggml_set_name(matmul, "pynq-smoke-matmul");

    if (!ggml_backend_supports_op(backend, copy)) {
        std::fprintf(stderr, "pynq backend does not support byte COPY\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    if (!ggml_backend_supports_op(backend, matmul)) {
        std::fprintf(stderr, "pynq backend does not support Q1A8 MUL_MAT\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx,
        ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "pynq buffer allocation failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> expected(n_values);
    for (int64_t i = 0; i < n_values; ++i) {
        expected[static_cast<size_t>(i)] = 0.25f + static_cast<float>(i) * 1.5f;
    }
    ggml_backend_tensor_set(tensor, expected.data(), 0, expected.size() * sizeof(float));

    std::vector<float> patch(view_values);
    for (int64_t i = 0; i < view_values; ++i) {
        patch[static_cast<size_t>(i)] = -100.0f - static_cast<float>(i);
        expected[static_cast<size_t>(view_start + i)] = patch[static_cast<size_t>(i)];
    }
    ggml_backend_tensor_set(view, patch.data(), 0, patch.size() * sizeof(float));

    std::vector<float> actual(n_values, 0.0f);
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    if (!same_floats(expected, actual)) {
        std::fprintf(stderr, "remote tensor upload/download did not round trip\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, copy);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "pynq backend RUN_GRAPH COPY failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> copied(view_values, 0.0f);
    ggml_backend_tensor_get(copy_dst_view, copied.data(), 0, copied.size() * sizeof(float));
    if (!same_floats(patch, copied)) {
        std::fprintf(stderr, "RUN_GRAPH COPY output did not round trip\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_set(
        matmul_src0,
        matmul_q1_weights.data(),
        0,
        matmul_q1_weights.size());
    ggml_backend_tensor_set(
        matmul_src1,
        matmul_acts.data(),
        0,
        matmul_acts.size() * sizeof(float));
    ggml_cgraph * matmul_graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(matmul_graph, matmul);
    if (ggml_backend_graph_compute(backend, matmul_graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "pynq backend RUN_GRAPH MATMUL_Q1A8 failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> matmul_actual(matmul_expected.size(), 0.0f);
    ggml_backend_tensor_get(
        matmul,
        matmul_actual.data(),
        0,
        matmul_actual.size() * sizeof(float));
    if (!same_floats(matmul_actual, matmul_expected, 1e-3f)) {
        std::fprintf(stderr, "RUN_GRAPH MATMUL_Q1A8 output mismatch\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    int scheduler_splits = 0;
    ggml_backend_load_all_from_path(PYNQ_GGML_BACKEND_DIR);
    ggml_backend_t cpu_backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_backend == nullptr) {
        std::fprintf(stderr, "ggml CPU backend init failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    if (!scheduler_matmul_smoke(
            backend,
            cpu_backend,
            matmul_q1_weights,
            matmul_acts,
            matmul_expected,
            &scheduler_splits)) {
        ggml_backend_free(cpu_backend);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(cpu_backend);

    std::printf(
        "pynq backend smoke ok: total=%zu free=%zu root_bytes=%zu view_bytes=%zu "
        "copy_bytes=%zu matmul_cells=%zu scheduler_splits=%d\n",
        total_bytes,
        free_bytes,
        expected.size() * sizeof(float),
        patch.size() * sizeof(float),
        copied.size() * sizeof(float),
        matmul_actual.size(),
        scheduler_splits);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
