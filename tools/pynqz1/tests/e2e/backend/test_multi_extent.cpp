#include "common.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

// Allocate, upload, then download a tensor large enough that the daemon
// allocator must span multiple slabs to satisfy it. Default daemon config
// (8 MiB heap, 1 MiB slabs) makes a 6 MiB tensor cross at least six slabs.
//
// The test mirrors what happens in production: ggml gives the backend ONE
// buffer that's larger than any single CMA chunk. If multi-extent wire
// marshaling regresses, this catches it before model load does.
bool run_multi_extent(ggml_backend_t backend, ggml_backend_dev_t dev) {
    using namespace pynq_e2e;
    constexpr int64_t n_floats = 6 * 1024 * 1024 / 4;  // 6 MiB of F32

    GGML_UNUSED(backend);

    const ggml_init_params params = {
        /* .mem_size   = */ 4 * ggml_tensor_overhead() + 4096,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_floats);
    ggml_set_name(t, "multi-extent");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(dev));
    if (buffer == nullptr) {
        std::fprintf(stderr, "multi_extent: buffer alloc failed — daemon may not have enough memory\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<float> source(n_floats);
    for (int64_t i = 0; i < n_floats; ++i) {
        source[static_cast<size_t>(i)] = static_cast<float>((i * 1103515245LL + 12345LL) & 0xFFFF) * 1e-3f;
    }
    ggml_backend_tensor_set(t, source.data(), 0, source.size() * sizeof(float));

    std::vector<float> actual(n_floats, 0.0f);
    ggml_backend_tensor_get(t, actual.data(), 0, actual.size() * sizeof(float));
    const bool ok = same_floats(source, actual);
    if (!ok) {
        std::fprintf(stderr, "multi_extent: round trip mismatch\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}
