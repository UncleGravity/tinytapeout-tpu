#include "matmul.h"
#include "transport.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

// Reference mirrors ggml_vec_dot_q1_0_q8_0: weights group at 128 with FP32 mean
// (== Q1_0's d_fp32 for true Q1_0 inputs), activations quantize at 32 with an
// FP16-roundtripped scale, accumulation is `acc += d_w * d_a * sub_sum` once
// per Q8_0 sub-block.
float expected_cell(const float * weights, const float * acts, int k) {
    constexpr int q1_group = 128;
    constexpr int q8_block = 32;

    auto fp16_roundtrip = [](float x) {
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    };

    float acc = 0.0f;
    for (int q1s = 0; q1s < k; q1s += q1_group) {
        const int q1e = std::min(q1s + q1_group, k);
        float w_abs_sum = 0.0f;
        for (int i = q1s; i < q1e; ++i) {
            w_abs_sum += std::fabs(weights[i]);
        }
        const float d_w = (q1e > q1s) ? w_abs_sum / (float) (q1e - q1s) : 0.0f;
        if (d_w == 0.0f) {
            continue;
        }

        for (int q8s = q1s; q8s < q1e; q8s += q8_block) {
            const int q8e = std::min(q8s + q8_block, q1e);

            float amax = 0.0f;
            for (int i = q8s; i < q8e; ++i) {
                if (std::isfinite(acts[i])) {
                    amax = std::max(amax, std::fabs(acts[i]));
                }
            }
            if (amax == 0.0f) {
                continue;
            }
            const float d_a = fp16_roundtrip(amax / 127.0f);
            const float id  = 127.0f / amax;

            int sub_sum = 0;
            for (int i = q8s; i < q8e; ++i) {
                if (!std::isfinite(acts[i])) {
                    continue;
                }
                long q = std::lround(acts[i] * id);
                if (q < -128) q = -128;
                if (q > 127)  q = 127;
                const int8_t a_q = (int8_t) q;
                sub_sum += (weights[i] >= 0.0f) ? (int) a_q : -(int) a_q;
            }
            acc += d_w * d_a * (float) sub_sum;
        }
    }

    return acc;
}

} // namespace

// Run a (rows x k) * (k x cols) matmul through the verilator transport and
// compare every output cell against the float reference. Returns true on
// match.
static bool run_case(int k, int rows, int cols,
                    bonsai::Transport & transport) {
    std::vector<uint8_t> memory((size_t) 1 << 22);
    ggml_init_params params;
    params.mem_size   = memory.size();
    params.mem_buffer = memory.data();
    params.no_alloc   = false;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "matmul-smoke: ggml_init failed for k=%d\n", k);
        return false;
    }

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cols);
    ggml_tensor * dst  = ggml_mul_mat(ctx, src0, src1);
    float * weights = (float *) src0->data;
    float * acts    = (float *) src1->data;

    // Deterministic pseudo-random {-1, +1} weights and small int activations.
    // Different seeds per row/col so each (row,col) pair sees a unique
    // dot product, including sign-flips spanning Q1 group boundaries.
    auto wbit = [](int row, int idx) {
        uint32_t h = (uint32_t) (row * 2654435761u) ^ (uint32_t) (idx * 374761393u);
        h ^= h >> 17;
        return (h & 1) ? 1.0f : -1.0f;
    };
    auto aval = [](int col, int idx) {
        uint32_t h = (uint32_t) (col * 1597334677u) ^ (uint32_t) (idx * 2246822519u);
        h ^= h >> 13;
        return (float) (int8_t) (h & 0x7f) * (col % 2 == 0 ? 1.0f : -1.0f) * 0.25f;
    };

    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < k; ++i) weights[row * k + i] = wbit(row, i);
    }
    for (int col = 0; col < cols; ++col) {
        for (int i = 0; i < k; ++i) acts[col * k + i] = aval(col, i);
    }

    bonsai::MatMulJob job;
    if (!bonsai::make_matmul_job(dst, &job)) {
        std::fprintf(stderr, "matmul-smoke: make_matmul_job rejected k=%d\n", k);
        ggml_free(ctx);
        return false;
    }
    if (!bonsai::run_bonsai_matmul(job, transport)) {
        std::fprintf(stderr, "matmul-smoke: run_bonsai_matmul failed via %s for k=%d\n",
                     transport.name(), k);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> w_row(k), a_col(k);
    const float * got = (const float *) dst->data;
    bool ok = true;
    for (int col = 0; col < cols; ++col) {
        for (int i = 0; i < k; ++i) a_col[i] = aval(col, i);
        for (int row = 0; row < rows; ++row) {
            for (int i = 0; i < k; ++i) w_row[i] = wbit(row, i);
            const float expected = expected_cell(w_row.data(), a_col.data(), k);
            const float actual   = got[col * rows + row];
            const float tol      = std::max(1e-4f, 1e-3f * std::fabs(expected));
            if (std::fabs(actual - expected) > tol) {
                std::fprintf(
                    stderr,
                    "matmul-smoke: mismatch k=%d row=%d col=%d got=%.6f expected=%.6f\n",
                    k, row, col, actual, expected);
                ok = false;
            }
        }
    }

    ggml_free(ctx);
    return ok;
}

int main() {
    std::unique_ptr<bonsai::Transport> transport = bonsai::create_verilator_transport();
    if (transport == nullptr) {
        std::fprintf(stderr, "bonsai-matmul-smoke: verilator transport unavailable "
                             "(was the build configured with -DBONSAI_ENABLE_VERILATOR=ON?)\n");
        return 1;
    }

    // Sweep K to exercise:
    //   k=4    smallest case (1 Q8 sub-block, 1 Q1 group) — degenerate
    //   k=64   2 Q8 sub-blocks within one Q1 group — exercises sub-block run boundaries
    //   k=160  spans 2 Q1 groups (5 Q8 sub-blocks total) — exercises group transitions
    // rows=3 trips the odd-row tail path (one paired (0,1), one unpaired (2)).
    bool ok = true;
    ok &= run_case(  4, 3, 2, *transport);
    ok &= run_case( 64, 3, 2, *transport);
    ok &= run_case(160, 3, 2, *transport);

    if (ok) {
        std::printf("bonsai-matmul-smoke: transport=%s passed\n", transport->name());
    }
    return ok ? 0 : 1;
}
