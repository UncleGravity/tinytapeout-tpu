#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

bool run_copy(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t n_values = 64;
    constexpr int64_t view_values = 8;
    constexpr int64_t view_start = 11;
    constexpr size_t graph_size = 4;

    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false) + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(src, "copy-src");
    ggml_tensor * src_view = ggml_view_1d(ctx, src, view_values, view_start * sizeof(float));
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(dst, "copy-dst");
    ggml_tensor * dst_view = ggml_view_1d(ctx, dst, view_values, view_start * sizeof(float));
    ggml_tensor * copy = ggml_cpy(ctx, src_view, dst_view);
    ggml_set_name(copy, "copy");

    if (!ggml_backend_supports_op(backend, copy)) {
        std::fprintf(stderr, "copy: backend does not support byte COPY\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "copy: buffer alloc failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> seed(n_values);
    for (int64_t i = 0; i < n_values; ++i) seed[static_cast<size_t>(i)] = static_cast<float>(i) - 16.0f;
    ggml_backend_tensor_set(src, seed.data(), 0, seed.size() * sizeof(float));

    std::vector<float> patch(view_values);
    for (int64_t i = 0; i < view_values; ++i) patch[static_cast<size_t>(i)] = -100.0f - static_cast<float>(i);
    ggml_backend_tensor_set(src_view, patch.data(), 0, patch.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(graph, copy);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (!computed) {
        std::fprintf(stderr, "copy: RUN_GRAPH COPY failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> copied(view_values, 0.0f);
    ggml_backend_tensor_get(dst_view, copied.data(), 0, copied.size() * sizeof(float));
    const bool ok = pynq_e2e::same_floats(patch, copied);
    if (!ok) {
        std::fprintf(stderr, "copy: COPY output mismatch\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}
