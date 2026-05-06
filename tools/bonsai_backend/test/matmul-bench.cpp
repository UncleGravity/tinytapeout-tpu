#include "matmul.h"
#include "transport.h"

#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

// End-to-end matmul timing via the USB transport. Goes through matmul.cpp's
// run_bonsai_matmul (the actual inference code path), so it picks up
// host-side overhead (dequant, quant, plan build, fold) on top of pure wire
// time. Use for measuring optimizations like activation hoisting that don't
// show up in transport-level bench.

static void run_one(bonsai::Transport & transport, int k, int m, int n,
                    int repeats, const char * label) {
    std::vector<uint8_t> memory((size_t) 1 << 27);  // 128 MiB
    ggml_init_params params;
    params.mem_size   = memory.size();
    params.mem_buffer = memory.data();
    params.no_alloc   = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "matmul-bench: ggml_init failed (k=%d m=%d n=%d)\n", k, m, n);
        return;
    }

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);

    float * weights = (float *) src0->data;
    float * acts    = (float *) src1->data;

    // Cheap deterministic patterns — don't matter for timing, but keep the
    // chip from short-circuiting on all-zero scales.
    for (int i = 0; i < k * m; ++i) weights[i] = (i % 3 == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < k * n; ++i) acts[i] = (float) ((i % 17) - 8);

    bonsai::MatMulJob job;
    if (!bonsai::make_matmul_job(dst, &job)) {
        std::fprintf(stderr, "matmul-bench: make_matmul_job rejected k=%d m=%d n=%d\n", k, m, n);
        ggml_free(ctx);
        return;
    }

    // Warmup once so any first-time scratch allocations land before the timer.
    if (!bonsai::run_bonsai_matmul(job, transport)) {
        std::fprintf(stderr, "matmul-bench: warmup run failed for %s\n", label);
        ggml_free(ctx);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        if (!bonsai::run_bonsai_matmul(job, transport)) {
            std::fprintf(stderr, "matmul-bench: timed run %d failed for %s\n", i, label);
            ggml_free(ctx);
            return;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const long long cell_pairs_per = ((long long) m + 1) / 2 * (long long) n;
    const long long cell_pairs = cell_pairs_per * (long long) repeats;
    std::printf(
        "matmul-bench: %-22s  k=%5d m=%5d n=%2d  %3d runs  %.1f ms total  %.3f ms/matmul  %.3f ms/cell-pair\n",
        label, k, m, n, repeats, ms, ms / repeats,
        ms / (double) cell_pairs);

    ggml_free(ctx);
}

int main() {
    auto transport = bonsai::create_usb_transport();
    if (transport == nullptr) {
        std::fprintf(stderr, "matmul-bench: USB transport unavailable (no chip on bus?)\n");
        return 1;
    }

    // Sweep representative shapes from Bonsai-1.7B (K, M, N):
    //   K=2048, M=2048, N=1: Q/O proj per decode token (1024 cell-pairs)
    //   K=2048, M=2048, N=2: same but T=2 prefill (2048 cell-pairs)
    //   K=2048, M=128,  N=1: small slice for low-noise per-cell timing
    //   K=6144, M=2048, N=1: down proj per decode token
    run_one(*transport, 2048,   128, 1,  10, "K2048 M128  N1");
    run_one(*transport, 2048,   128, 2,  10, "K2048 M128  N2");
    run_one(*transport, 2048,  2048, 1,   3, "K2048 M2048 N1");
    run_one(*transport, 6144,   128, 1,   5, "K6144 M128  N1");
    return 0;
}
