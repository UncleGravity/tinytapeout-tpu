#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <memory>
#include <vector>

bool run_upload_download(ggml_backend_t backend, ggml_backend_dev_t dev) {
    constexpr int64_t n_values = 64;
    constexpr int64_t view_values = 8;
    constexpr int64_t view_start = 11;
    GGML_UNUSED(backend);

    const ggml_init_params params = {
        /* .mem_size   = */ 8 * ggml_tensor_overhead() + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_values);
    ggml_set_name(tensor, "upload-download-root");
    ggml_tensor * view = ggml_view_1d(ctx, tensor, view_values, view_start * sizeof(float));
    ggml_set_name(view, "upload-download-view");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "upload_download: buffer alloc failed\n");
        ggml_free(ctx);
        return false;
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
    const bool ok = pynq_e2e::same_floats(expected, actual);
    if (!ok) {
        std::fprintf(stderr, "upload_download: round trip mismatch\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}
