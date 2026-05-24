#include "common.h"

#include "ggml-pynq.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>

namespace {

struct DirectTest {
    const char * name;
    bool (*fn)(ggml_backend_t, ggml_backend_dev_t);
};

struct SchedTest {
    const char * name;
    bool (*fn)(ggml_backend_t, ggml_backend_t);
};

constexpr DirectTest k_direct_tests[] = {
    { "upload_download",  run_upload_download },
    { "copy",             run_copy },
    { "matmul_direct",    run_matmul_direct },
    { "swiglu_direct",    run_swiglu_direct },
    { "multi_extent",     run_multi_extent },
};

constexpr SchedTest k_sched_tests[] = {
    { "matmul_scheduler", run_matmul_scheduler },
    { "glue_scheduler",   run_glue_scheduler },
    { "swiglu_scheduler", run_swiglu_scheduler },
};

} // namespace

int main() {
    ggml_backend_reg_t reg = ggml_backend_pynq_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    if (dev == nullptr) {
        std::fprintf(stderr, "pynq backend did not register a device\n");
        return 1;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
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

    int passed = 0;
    int failed = 0;

    for (const auto & t : k_direct_tests) {
        const bool ok = t.fn(backend, dev);
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", t.name);
        ok ? ++passed : ++failed;
    }

    ggml_backend_load_all_from_path(PYNQ_GGML_BACKEND_DIR);
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_backend == nullptr) {
        std::fprintf(stderr, "ggml CPU backend init failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    for (const auto & t : k_sched_tests) {
        const bool ok = t.fn(backend, cpu_backend);
        std::printf("[%s] %s\n", ok ? "ok" : "FAIL", t.name);
        ok ? ++passed : ++failed;
    }
    ggml_backend_free(cpu_backend);
    ggml_backend_free(backend);

    std::printf("pynq-backend-smoke: %d passed, %d failed (total=%zu free=%zu)\n",
        passed, failed, total_bytes, free_bytes);
    return failed == 0 ? 0 : 1;
}
