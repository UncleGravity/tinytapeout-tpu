#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

// One scheduler-driven graph exercising add + scale + rms_norm + silu + mul.
// If any glue op falls back to CPU, the assertion on tensor backend below
// will catch it before the numerical check runs.
bool run_glue_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    using namespace pynq_e2e;
    constexpr size_t graph_size = 16;
    auto input_data = make_glue_input();
    auto bias_data = make_glue_bias();
    auto expected = expected_glue_output(input_data, bias_data);

    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_glue_rows, k_glue_cols);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k_glue_rows);
    ggml_tensor * add = ggml_add(ctx, input, bias);
    ggml_tensor * scale = ggml_scale(ctx, add, 0.5f);
    ggml_tensor * norm = ggml_rms_norm(ctx, scale, 1.0e-6f);
    ggml_tensor * act = ggml_silu(ctx, norm);
    ggml_tensor * out = ggml_mul(ctx, act, bias);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g, out);

    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
    if (sched == nullptr || !ggml_backend_sched_alloc_graph(sched, g)) {
        std::fprintf(stderr, "glue_scheduler: alloc failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched, out) != backend) {
        std::fprintf(stderr, "glue_scheduler: output not assigned to PYNQ\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "glue_scheduler: compute failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(actual, expected, 1e-5f);
    if (!ok) std::fprintf(stderr, "glue_scheduler: output mismatch\n");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}
