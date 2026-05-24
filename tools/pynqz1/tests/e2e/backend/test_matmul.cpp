#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

namespace {

bool prepare_matmul_inputs(std::vector<uint8_t> * q1_weights,
                           std::vector<float> * acts,
                           std::vector<float> * expected) {
    std::vector<float> weights = pynq_e2e::make_matmul_weights();
    *acts = pynq_e2e::make_matmul_acts();
    *expected = pynq_e2e::expected_matmul(weights, *acts);
    return pynq_e2e::quantize_matmul_weights(weights, q1_weights);
}

} // namespace

bool run_matmul_direct(ggml_backend_t backend, ggml_backend_dev_t dev) {
    using namespace pynq_e2e;
    constexpr size_t graph_size = 4;
    std::vector<uint8_t> q1_weights;
    std::vector<float> acts;
    std::vector<float> expected;
    if (!prepare_matmul_inputs(&q1_weights, &acts, &expected)) {
        std::fprintf(stderr, "matmul_direct: quantization failed\n");
        return false;
    }

    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, k_matmul_k, k_matmul_rows);
    ggml_set_name(w, "matmul-weights");
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_matmul_k, k_matmul_cols);
    ggml_set_name(a, "matmul-acts");
    ggml_tensor * out = ggml_mul_mat(ctx, w, a);
    ggml_set_name(out, "matmul");

    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "matmul_direct: backend does not support Q1A8 MUL_MAT\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "matmul_direct: buffer alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(w, q1_weights.data(), 0, q1_weights.size());
    ggml_backend_tensor_set(a, acts.data(), 0, acts.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g, out);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "matmul_direct: RUN_GRAPH MATMUL_Q1A8 failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(actual, expected, 1e-3f);
    if (!ok) std::fprintf(stderr, "matmul_direct: output mismatch\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool run_matmul_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    using namespace pynq_e2e;
    constexpr size_t graph_size = 4;
    std::vector<uint8_t> q1_weights;
    std::vector<float> acts;
    std::vector<float> expected;
    if (!prepare_matmul_inputs(&q1_weights, &acts, &expected)) {
        std::fprintf(stderr, "matmul_scheduler: quantization failed\n");
        return false;
    }

    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q1_0, k_matmul_k, k_matmul_rows);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_matmul_k, k_matmul_cols);
    ggml_tensor * out = ggml_mul_mat(ctx, w, a);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g, out);

    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
    if (sched == nullptr || !ggml_backend_sched_alloc_graph(sched, g)) {
        std::fprintf(stderr, "matmul_scheduler: alloc failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched, out) != backend) {
        std::fprintf(stderr, "matmul_scheduler: MUL_MAT not assigned to PYNQ\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(w, q1_weights.data(), 0, q1_weights.size());
    ggml_backend_tensor_set(a, acts.data(), 0, acts.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "matmul_scheduler: compute failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(actual, expected, 1e-3f);
    if (!ok) std::fprintf(stderr, "matmul_scheduler: output mismatch\n");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}
