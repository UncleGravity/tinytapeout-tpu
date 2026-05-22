#include "ggml-pynq.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

bool same_floats(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > 1e-6f) {
            return false;
        }
    }
    return true;
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

    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() + 4096,
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

    std::printf(
        "pynq backend smoke ok: total=%zu free=%zu root_bytes=%zu view_bytes=%zu\n",
        total_bytes,
        free_bytes,
        expected.size() * sizeof(float),
        patch.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
