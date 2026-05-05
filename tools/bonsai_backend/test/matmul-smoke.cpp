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

int main() {
    constexpr int k = 4;
    constexpr int rows = 3;
    constexpr int cols = 2;

    std::vector<uint8_t> memory(1 << 20);
    ggml_init_params params;
    params.mem_size = memory.size();
    params.mem_buffer = memory.data();
    params.no_alloc = false;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "bonsai-matmul-smoke: ggml_init failed\n");
        return 1;
    }

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, cols);
    ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);

    float * weights = (float *) src0->data;
    float * acts = (float *) src1->data;
    const float weights_by_row[rows][k] = {
        {  1.0f, -1.0f,  1.0f, -1.0f },
        { -1.0f, -1.0f,  1.0f,  1.0f },
        {  1.0f,  1.0f,  1.0f, -1.0f },
    };
    const float acts_by_col[cols][k] = {
        {  3.0f, -2.0f,  5.0f,  7.0f },
        { -4.0f,  6.0f,  1.0f, -3.0f },
    };

    for (int row = 0; row < rows; ++row) {
        for (int i = 0; i < k; ++i) {
            weights[row * k + i] = weights_by_row[row][i];
        }
    }
    for (int col = 0; col < cols; ++col) {
        for (int i = 0; i < k; ++i) {
            acts[col * k + i] = acts_by_col[col][i];
        }
    }

    bonsai::MatMulJob job;
    if (!bonsai::make_matmul_job(dst, &job)) {
        std::fprintf(stderr, "bonsai-matmul-smoke: make_matmul_job rejected the smoke graph\n");
        ggml_free(ctx);
        return 1;
    }

    // Verilator-driven RTL parity: lowering layer + actual gates, end-to-end.
    std::unique_ptr<bonsai::Transport> transport = bonsai::create_verilator_transport();
    if (transport == nullptr) {
        std::fprintf(stderr, "bonsai-matmul-smoke: verilator transport unavailable "
                             "(was the build configured with -DBONSAI_ENABLE_VERILATOR=ON?)\n");
        ggml_free(ctx);
        return 1;
    }

    if (!bonsai::run_bonsai_matmul(job, *transport)) {
        std::fprintf(stderr, "bonsai-matmul-smoke: run_bonsai_matmul failed via %s\n", transport->name());
        ggml_free(ctx);
        return 1;
    }

    const float * got = (const float *) dst->data;
    bool ok = true;
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            const float expected = expected_cell(weights_by_row[row], acts_by_col[col], k);
            const float actual = got[col * rows + row];
            if (std::fabs(actual - expected) > 1e-5f) {
                std::fprintf(
                    stderr,
                    "bonsai-matmul-smoke: mismatch row=%d col=%d got=%.6f expected=%.6f\n",
                    row,
                    col,
                    actual,
                    expected);
                ok = false;
            }
        }
    }

    if (ok) {
        std::printf("bonsai-matmul-smoke: transport=%s passed\n", transport->name());
    }

    ggml_free(ctx);
    return ok ? 0 : 1;
}
