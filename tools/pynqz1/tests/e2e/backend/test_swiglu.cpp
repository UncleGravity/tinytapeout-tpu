#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

bool run_swiglu_direct(ggml_backend_t backend, ggml_backend_dev_t dev) {
    using namespace pynq_e2e;
    constexpr size_t graph_size = 4;
    auto gate = make_glue_input();
    auto up = make_swiglu_up();
    auto expected = expected_swiglu_output(gate, up);

    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * g0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_glue_rows, k_glue_cols);
    ggml_tensor * g1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_glue_rows, k_glue_cols);
    ggml_tensor * out = ggml_swiglu_split(ctx, g0, g1);

    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "swiglu_direct: backend does not support split SwiGLU\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "swiglu_direct: buffer alloc failed\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(g0, gate.data(), 0, gate.size() * sizeof(float));
    ggml_backend_tensor_set(g1, up.data(), 0, up.size() * sizeof(float));

    ggml_cgraph * g = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g, out);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "swiglu_direct: compute failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(actual, expected, 1e-5f);
    if (!ok) std::fprintf(stderr, "swiglu_direct: output mismatch\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

bool run_swiglu_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend) {
    using namespace pynq_e2e;
    constexpr size_t graph_size = 4;
    auto gate = make_glue_input();
    auto up = make_swiglu_up();
    auto expected = expected_swiglu_output(gate, up);

    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * g0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_glue_rows, k_glue_cols);
    ggml_tensor * g1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_glue_rows, k_glue_cols);
    ggml_tensor * out = ggml_swiglu_split(ctx, g0, g1);
    ggml_cgraph * g = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(g, out);

    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
    if (sched == nullptr || !ggml_backend_sched_alloc_graph(sched, g)) {
        std::fprintf(stderr, "swiglu_scheduler: alloc failed\n");
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_sched_get_tensor_backend(sched, out) != backend) {
        std::fprintf(stderr, "swiglu_scheduler: SwiGLU not assigned to PYNQ\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(g0, gate.data(), 0, gate.size() * sizeof(float));
    ggml_backend_tensor_set(g1, up.data(), 0, up.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, g) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "swiglu_scheduler: compute failed\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }
    std::vector<float> actual(expected.size(), 0.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(actual, expected, 1e-5f);
    if (!ok) std::fprintf(stderr, "swiglu_scheduler: output mismatch\n");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}
