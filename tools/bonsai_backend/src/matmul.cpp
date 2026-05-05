#include "matmul.h"

#include "plan.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace bonsai {

namespace {

static bool type_can_convert_to_float(ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->to_float != nullptr;
}

static const float * row_to_float(
        const ggml_tensor * tensor,
        const void * data,
        i64 n,
        std::vector<float> & scratch) {
    if (tensor->type == GGML_TYPE_F32) {
        return (const float *) data;
    }

    scratch.resize((size_t) n);
    ggml_get_type_traits(tensor->type)->to_float(data, scratch.data(), n);
    return scratch.data();
}

// Layouts mirror ggml's reference: weights group at 128 elements (Q1_0 / QK1_0),
// activations quantize at 32 elements (Q8_0 / QK8_0). The vec_dot in
// ggml_vec_dot_q1_0_q8_0 walks Q1_0 blocks outside, Q8_0 sub-blocks inside, and
// accumulates `d_w * d_a * sumi` once per Q8_0 sub-block. We mirror that order
// so the output matches the CPU reference up to FP rounding noise.
constexpr i64 k_weight_group = 128;
constexpr i64 k_act_block    = 32;

// Per-group weight scale alpha_w = mean(|w|) over [start, end). For Q1_0 the
// dequantized weights are exactly +/- d_fp32 (with d_fp32 = FP16->FP32 of the
// stored half scale), so mean(|w|) == d_fp32 bit-exactly.
static float weight_group_scale(const float * w, i64 start, i64 end) {
    if (end <= start) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (i64 i = start; i < end; ++i) {
        sum += std::fabs(w[i]);
    }
    return sum / (float) (end - start);
}

// Q8_0 layout: one int8 per element plus one FP16-rounded scale per 32-element
// block. Mirrors quantize_row_q8_0_ref so the int8 stream and the scale stream
// match the values that ggml_vec_dot_q1_0_q8_0 sees on the CPU path.
struct ActQuants {
    std::vector<int8_t> q;
    std::vector<float>  scales;
};

static void quantize_acts_q8_0(const float * a, i64 n, ActQuants & out) {
    out.q.resize((size_t) n);
    const i64 n_blocks = (n + k_act_block - 1) / k_act_block;
    out.scales.resize((size_t) n_blocks);

    for (i64 b = 0; b < n_blocks; ++b) {
        const i64 b_start = b * k_act_block;
        const i64 b_end   = std::min(b_start + k_act_block, n);

        float amax = 0.0f;
        for (i64 i = b_start; i < b_end; ++i) {
            if (std::isfinite(a[i])) {
                amax = std::max(amax, std::fabs(a[i]));
            }
        }

        if (amax == 0.0f) {
            out.scales[(size_t) b] = 0.0f;
            for (i64 i = b_start; i < b_end; ++i) {
                out.q[(size_t) i] = 0;
            }
            continue;
        }

        const float d  = amax / 127.0f;
        const float id = 1.0f / d;
        // Roundtrip the scale through FP16 to match Q8_0's stored precision —
        // that is the value the CPU vec_dot multiplies into the final sum.
        out.scales[(size_t) b] = GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(d));
        for (i64 i = b_start; i < b_end; ++i) {
            if (!std::isfinite(a[i])) {
                out.q[(size_t) i] = 0;
                continue;
            }
            long q = std::lround(a[i] * id);
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            out.q[(size_t) i] = (int8_t) q;
        }
    }
}

struct CellScratch {
    std::vector<float>   w_floats;
    std::vector<float>   a_floats;
    ActQuants            act_quants;
    // Reusable Plan + matching output buffer; cleared and re-filled per
    // Q8 sub-block so we keep their backing storage hot across cells.
    Plan                 plan;
    std::vector<int16_t> psums_buf;
};

static bool run_cell(
        const MatMulJob & job,
        i64 linear_index,
        Transport & transport,
        CellScratch & scratch) {
    const i64 row = linear_index % job.n_rows;
    const i64 rem = linear_index / job.n_rows;

    const i64 b3  = rem / (job.n_b2 * job.n_cols);
    const i64 b2  = (rem - b3 * job.n_b2 * job.n_cols) / job.n_cols;
    const i64 col = rem - b3 * job.n_b2 * job.n_cols - b2 * job.n_cols;

    const i64 src0_i3 = b3 / job.src0_b3;
    const i64 src0_i2 = b2 / job.src0_b2;

    const void * src0_row =
        (const char *) job.src0->data +
        row * job.src0_nb1 +
        src0_i2 * job.src0_nb2 +
        src0_i3 * job.src0_nb3;

    const void * src1_row =
        (const char *) job.src1->data +
        col * job.src1_nb1 +
        b2 * job.src1_nb2 +
        b3 * job.src1_nb3;

    float * dst_cell =
        (float *) ((char *) job.dst->data +
        row * job.dst_nb0 +
        col * job.dst_nb1 +
        b2 * job.dst_nb2 +
        b3 * job.dst_nb3);

    const float * weights = row_to_float(job.src0, src0_row, job.k, scratch.w_floats);
    const float * acts    = row_to_float(job.src1, src1_row, job.k, scratch.a_floats);

    quantize_acts_q8_0(acts, job.k, scratch.act_quants);

    // Mirror ggml_vec_dot_q1_0_q8_0's accumulation order: outer loop over Q1_0
    // weight blocks (128 elements), inner loop over Q8_0 sub-blocks (32 acts).
    // Each Q8_0 sub-block sums in int32 with no FP error (max magnitude
    // 32 * 127 = 4064, fits in int16 already), then a single
    // `acc += d_w * d_a * sub_sum` fold per sub-block matches the reference
    // exactly modulo FP add-order across blocks.
    float acc = 0.0f;
    for (i64 q1_start = 0; q1_start < job.k; q1_start += k_weight_group) {
        const i64 q1_end = std::min(q1_start + k_weight_group, job.k);
        const float d_w = weight_group_scale(weights, q1_start, q1_end);
        if (d_w == 0.0f) {
            continue;
        }

        for (i64 q8_start = q1_start; q8_start < q1_end; q8_start += k_act_block) {
            const i64 q8_end = std::min(q8_start + k_act_block, q1_end);
            const i64 a_block = q8_start / k_act_block;
            const float d_a = scratch.act_quants.scales[(size_t) a_block];
            if (d_a == 0.0f) {
                continue;
            }

            const int n_tiles = (int) ((q8_end - q8_start + Tile::cols - 1) / Tile::cols);
            scratch.plan.clear();
            scratch.plan.ops.reserve((size_t) n_tiles);
            scratch.psums_buf.resize((size_t) n_tiles);

            for (int t = 0; t < n_tiles; ++t) {
                const i64 k0 = q8_start + (i64) t * Tile::cols;
                uint8_t packed_weights = 0;
                int8_t acts_buf[Tile::cols];
                for (int lane = 0; lane < Tile::cols; ++lane) {
                    const i64 k = k0 + lane;
                    const bool weight_bit = k < q8_end && weights[k] >= 0.0f;
                    packed_weights |= (uint8_t) weight_bit << lane;
                    acts_buf[lane] =
                        k < q8_end ? scratch.act_quants.q[(size_t) k] : (int8_t) 0;
                }
                // seed = 0: this Q8 sub-block accumulates fresh; the
                // per-block partial sum is folded into FP outside the
                // chip via `d_w * d_a * sub_sum` below.
                scratch.plan.add_matmul_tile(packed_weights, acts_buf, 0);
            }

            if (!transport.execute(scratch.plan, scratch.psums_buf.data())) {
                return false;
            }

            int32_t sub_sum = 0;
            for (int t = 0; t < n_tiles; ++t) {
                sub_sum += (int32_t) scratch.psums_buf[(size_t) t];
            }
            acc += d_w * d_a * (float) sub_sum;
        }
    }

    *dst_cell = acc;
    return true;
}

} // namespace

bool make_matmul_job(ggml_tensor * dst, MatMulJob * job) {
    if (dst == nullptr || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    if (!type_can_convert_to_float(src0->type) ||
            !type_can_convert_to_float(src1->type) ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] != src1->ne[0]) {
        return false;
    }

    if (src0->ne[0] % ggml_blck_size(src0->type) != 0 ||
            src1->ne[0] % ggml_blck_size(src1->type) != 0) {
        return false;
    }

    if (dst->ne[0] != src0->ne[1] ||
            dst->ne[1] != src1->ne[1] ||
            dst->ne[2] != src1->ne[2] ||
            dst->ne[3] != src1->ne[3]) {
        return false;
    }

    if (src0->ne[2] <= 0 ||
            src0->ne[3] <= 0 ||
            src1->ne[2] % src0->ne[2] != 0 ||
            src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }

    if (src0->nb[0] != ggml_type_size(src0->type) ||
            src1->nb[0] != ggml_type_size(src1->type) ||
            dst->nb[0] != sizeof(float)) {
        return false;
    }

    if (job == nullptr) {
        return true;
    }

    *job = {
        /* .dst      = */ dst,
        /* .src0     = */ src0,
        /* .src1     = */ src1,
        /* .k        = */ src0->ne[0],
        /* .n_rows   = */ dst->ne[0],
        /* .n_cols   = */ dst->ne[1],
        /* .n_b2     = */ dst->ne[2],
        /* .n_b3     = */ dst->ne[3],
        /* .src0_b2  = */ dst->ne[2] / src0->ne[2],
        /* .src0_b3  = */ dst->ne[3] / src0->ne[3],
        /* .src0_nb1 = */ src0->nb[1],
        /* .src0_nb2 = */ src0->nb[2],
        /* .src0_nb3 = */ src0->nb[3],
        /* .src1_nb1 = */ src1->nb[1],
        /* .src1_nb2 = */ src1->nb[2],
        /* .src1_nb3 = */ src1->nb[3],
        /* .dst_nb0  = */ dst->nb[0],
        /* .dst_nb1  = */ dst->nb[1],
        /* .dst_nb2  = */ dst->nb[2],
        /* .dst_nb3  = */ dst->nb[3],
    };

    return true;
}

bool supports_matmul(const ggml_tensor * op) {
    return make_matmul_job((ggml_tensor *) op, nullptr);
}

bool run_bonsai_matmul(const MatMulJob & job, Transport & transport) {
    const i64 work = job.n_rows * job.n_cols * job.n_b2 * job.n_b3;
    if (work == 0) {
        return true;
    }

    // Serial execution. The chip is shared across all backend instances in
    // a process and the per-X-frame mutex is finer-grained than a cell, so
    // parallel cells would interleave and corrupt chip state. The
    // verilator transport could parallelize cleanly (its state is
    // per-instance) but tests run on small inputs where the overhead is
    // not worth the complexity. Add a Transport::clone() pattern here when
    // a workload actually justifies it.
    CellScratch scratch;
    for (i64 i = 0; i < work; ++i) {
        if (!run_cell(job, i, transport, scratch)) {
            return false;
        }
    }

    // Per-tile error checks were dropped from the transport for batching
    // speed; sample the chip's error_latched bit once at end-of-matmul.
    return !(transport.status() & status_error);
}

} // namespace bonsai
