#include "internal.h"
#include "trace.h"

#include "proto/ops.h"

#include "ggml-backend-impl.h"

#include <cstring>

namespace pynq {

namespace P = pynq::proto;

// -- lowering helpers shared across op handlers ---------------------------

namespace {

struct F32BinaryLowering {
    const ggml_tensor * src0 = nullptr;
    const ggml_tensor * src1 = nullptr;
    bool src1_broadcast = false;
};

bool is_metadata_op(ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

bool supports_raw_copy(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_CPY) {
        return false;
    }
    const ggml_tensor * src = op->src[0];
    const ggml_tensor * dst = op->src[1];
    return src != nullptr &&
        dst != nullptr &&
        src->type == dst->type &&
        ggml_is_contiguous(src) &&
        ggml_is_contiguous(dst) &&
        ggml_nbytes(src) == ggml_nbytes(dst);
}

bool same_shape(const ggml_tensor * lhs, const ggml_tensor * rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs->ne[dim] != rhs->ne[dim]) {
            return false;
        }
    }
    return true;
}

bool is_contiguous_f32(const ggml_tensor * tensor) {
    return tensor != nullptr &&
        tensor->type == GGML_TYPE_F32 &&
        tensor->ne[0] > 0 &&
        ggml_nelements(tensor) > 0 &&
        ggml_is_contiguous(tensor);
}

int64_t flattened_cols(const ggml_tensor * tensor) {
    return ggml_nelements(tensor) / tensor->ne[0];
}

float op_param_f32(const ggml_tensor * tensor, int index) {
    float value = 0.0f;
    std::memcpy(
        &value,
        reinterpret_cast<const char *>(tensor->op_params) + index * sizeof(value),
        sizeof(value));
    return value;
}

int32_t op_param_i32(const ggml_tensor * tensor, int index) {
    int32_t value = 0;
    std::memcpy(
        &value,
        reinterpret_cast<const char *>(tensor->op_params) + index * sizeof(value),
        sizeof(value));
    return value;
}

bool is_row_broadcast_for(const ggml_tensor * src, const ggml_tensor * dst) {
    if (!is_contiguous_f32(src) || !is_contiguous_f32(dst) || src->ne[0] != dst->ne[0]) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (src->ne[dim] != 1) {
            return false;
        }
    }
    return true;
}

bool get_f32_binary_lowering(const ggml_tensor * op, F32BinaryLowering * lowering) {
    if (op == nullptr ||
        (op->op != GGML_OP_ADD && op->op != GGML_OP_MUL) ||
        !is_contiguous_f32(op)) {
        return false;
    }
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!is_contiguous_f32(src0) || !is_contiguous_f32(src1)) {
        return false;
    }
    if (same_shape(src0, op) && same_shape(src1, op)) {
        *lowering = F32BinaryLowering { src0, src1, false };
        return true;
    }
    if (same_shape(src0, op) && is_row_broadcast_for(src1, op)) {
        *lowering = F32BinaryLowering { src0, src1, true };
        return true;
    }
    if (same_shape(src1, op) && is_row_broadcast_for(src0, op)) {
        *lowering = F32BinaryLowering { src1, src0, true };
        return true;
    }
    return false;
}

bool supports_f32_binary(const ggml_tensor * op) {
    F32BinaryLowering lowering;
    return get_f32_binary_lowering(op, &lowering);
}

bool supports_scale_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_SCALE &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

bool supports_silu_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_UNARY &&
        ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

bool supports_swiglu_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_GLU &&
        ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        is_contiguous_f32(op->src[1]) &&
        same_shape(op, op->src[0]) &&
        same_shape(op, op->src[1]);
}

bool supports_rms_norm_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_RMS_NORM &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

// ROPE op_params layout (ggml.c ggml_rope_impl, 15 × i32):
//   [0]  legacy n_past (always 0)
//   [1]  n_dims
//   [2]  mode
//   [3]  legacy n_ctx (always 0)
//   [4]  n_ctx_orig
//   [5]  freq_base   (float bits)
//   [6]  freq_scale  (float bits)
//   [7]  ext_factor  (float bits) — YaRN extrapolation strength
//   [8]  attn_factor (float bits) — YaRN mscale base
//   [9]  beta_fast   (float bits) — YaRN ramp inner bound
//   [10] beta_slow   (float bits) — YaRN ramp outer bound
//
// Bonsai-1.7B uses YaRN scaling (ext_factor=1, freq_scale=0.25, beta_fast=32),
// so the kernel implements the full rope_yarn formula. The predicate only
// rejects unsupported shapes/dtypes/modes and the freq_factors input.
bool supports_rope_f32(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_ROPE) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src  = op->src[0];
    const ggml_tensor * pos  = op->src[1];
    const ggml_tensor * freq = (GGML_MAX_SRC >= 3) ? op->src[2] : nullptr;
    if (src == nullptr || pos == nullptr) return false;
    if (src->type != GGML_TYPE_F32) return false;
    if (pos->type != GGML_TYPE_I32) return false;
    if (freq != nullptr) return false;  // freq_factors (phi-3-128k) not supported
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) return false;
    if (src->ne[3] != 1 || op->ne[3] != 1) return false;
    if (src->ne[0] != op->ne[0] || src->ne[1] != op->ne[1] ||
        src->ne[2] != op->ne[2]) return false;
    const int32_t n_dims = op_param_i32(op, 1);
    const int32_t mode   = op_param_i32(op, 2);
    if (n_dims <= 0 || (n_dims & 1) || n_dims > src->ne[0]) return false;
    if (mode != 0 /*NORMAL*/ && mode != 2 /*NEOX*/) return false;
    return true;
}

// FLASH_ATTN_EXT op_params layout (ggml.c ggml_flash_attn_ext_impl):
//   [0] (f32) scale
//   [1] (f32) max_bias
//   [2] (f32) logit_softcap
//
// Bonsai-1.7B (Qwen3-based) uses: Q=F32, K=F16, V=F16, mask=F16, GQA,
// max_bias=0 (no ALiBi), logit_softcap=0, no sinks. v1 supports exactly
// that shape; everything else falls back to CPU.
//
// K and V tensors are normally views into the persistent KV cache, so
// they're regular-strided per dim but NOT ggml_is_contiguous in the
// conventional sense (head stride = head_dim * ctx_max, not
// head_dim * n_kv). The kernel takes nb1/nb2 strides explicitly.
bool supports_flash_attn_ext_f32(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_FLASH_ATTN_EXT) return false;
    if (op->type != GGML_TYPE_F32) return false;

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = (GGML_MAX_SRC >= 4) ? op->src[3] : nullptr;
    const ggml_tensor * sinks = (GGML_MAX_SRC >= 5) ? op->src[4] : nullptr;
    if (q == nullptr || k == nullptr || v == nullptr) return false;

    if (q->type != GGML_TYPE_F32) return false;
    if (k->type != GGML_TYPE_F16) return false;
    if (v->type != GGML_TYPE_F16) return false;
    if (mask != nullptr && mask->type != GGML_TYPE_F16) return false;
    if (sinks != nullptr) return false;  // not implemented

    // ggml_flash_attn_ext_impl shape conventions:
    //   q  : [head_dim_q, n_token, n_head,    1]
    //   k  : [head_dim_q, n_kv,    n_head_kv, 1]
    //   v  : [head_dim_v, n_kv,    n_head_kv, 1]
    //   dst: [head_dim_v, n_head,  n_token,   1]   (note: permuted)
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || op->ne[3] != 1) return false;
    if (q->ne[0] != k->ne[0]) return false;             // DK matches
    if (q->ne[0] <= 0 || v->ne[0] <= 0) return false;
    if (k->ne[1] != v->ne[1]) return false;             // n_kv matches
    if (k->ne[2] != v->ne[2]) return false;             // n_head_kv matches
    if (q->ne[2] <= 0 || k->ne[2] <= 0) return false;
    if (q->ne[2] % k->ne[2] != 0) return false;         // GQA divides evenly

    // nb0 must be the raw element size — this is the assertion ggml itself
    // makes (ggml/src/ggml-cpu/ops.cpp:8199-8201). If it ever fails we
    // need a deeper rethink; just reject for safety.
    if (q->nb[0] != ggml_type_size(GGML_TYPE_F32)) return false;
    if (k->nb[0] != ggml_type_size(GGML_TYPE_F16)) return false;
    if (v->nb[0] != ggml_type_size(GGML_TYPE_F16)) return false;
    if (mask != nullptr && mask->nb[0] != ggml_type_size(GGML_TYPE_F16)) return false;

    float max_bias = 0.0f, logit_softcap = 0.0f;
    std::memcpy(&max_bias,      reinterpret_cast<const char *>(op->op_params) + 1*sizeof(float), sizeof(float));
    std::memcpy(&logit_softcap, reinterpret_cast<const char *>(op->op_params) + 2*sizeof(float), sizeof(float));
    if (max_bias != 0.0f) return false;       // ALiBi not implemented
    if (logit_softcap != 0.0f) return false;  // softcap not implemented

    // Runtime safety cap — keeps KV cache footprint in the CMA budget.
    if (k->ne[1] > 8192) return false;
    return true;
}

// GET_ROWS: src0 has rows of some type (Q1_0 / F16 / F32), src1 is i32
// indices, dst is F32. Used for the token embedding lookup
// (Bonsai's tok_embd.weight is Q1_0 of shape [n_embd, vocab]).
bool supports_get_rows(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_GET_ROWS) return false;
    if (op->type != GGML_TYPE_F32) return false;
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) return false;
    if (src1->type != GGML_TYPE_I32) return false;
    if (src0->type != GGML_TYPE_Q1_0 &&
        src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_F32) {
        return false;
    }
    // Row count must align with Q1_0 block when src0 is Q1_0.
    if (src0->type == GGML_TYPE_Q1_0) {
        if (src0->ne[0] % ggml_blck_size(GGML_TYPE_Q1_0) != 0) {
            return false;
        }
        if (src0->ne[2] != 1 || src0->ne[3] != 1 || !ggml_is_contiguous(src0)) {
            return false;
        }
    }
    if (!ggml_is_contiguous(op)) return false;
    return true;
}

// SET_ROWS: writes src0 (F32) rows into dst (F16) at indices given by
// src1 (i32 or i64). Used by llama.cpp to append the new K/V into the
// KV cache each token (the dst is the F16 KV cache view).
bool supports_set_rows(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_SET_ROWS) return false;
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) return false;
    if (src0->type != GGML_TYPE_F32) return false;
    if (op->type != GGML_TYPE_F16) return false;
    if (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64) return false;
    if (src0->nb[0] != ggml_type_size(GGML_TYPE_F32)) return false;
    if (op->nb[0] != ggml_type_size(GGML_TYPE_F16)) return false;
    if (src0->ne[0] != op->ne[0]) return false;
    if (op->ne[2] % src1->ne[1] != 0) return false;
    if (op->ne[3] % src1->ne[2] != 0) return false;
    return true;
}

bool supports_matmul_q1a8(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * acts = op->src[1];
    if (weights == nullptr ||
        acts == nullptr ||
        weights->type != GGML_TYPE_Q1_0 ||
        acts->type != GGML_TYPE_F32 ||
        weights->ne[0] != acts->ne[0] ||
        weights->ne[0] <= 0 ||
        weights->ne[0] % ggml_blck_size(GGML_TYPE_Q1_0) != 0 ||
        op->ne[0] != weights->ne[1] ||
        op->ne[1] != acts->ne[1]) {
        return false;
    }
    for (int dim = 2; dim < GGML_MAX_DIMS; ++dim) {
        if (weights->ne[dim] != 1 || acts->ne[dim] != 1 || op->ne[dim] != 1) {
            return false;
        }
    }
    return ggml_is_contiguous(weights) && ggml_is_contiguous(acts) && ggml_is_contiguous(op);
}

// -- append_X_op ----------------------------------------------------------
//
// Each appender:
//   1. checks bindings + ranges
//   2. pushes one JSON op into ``ops``
//   3. pushes the destination handle into ``outputs``
// Errors are reported via GGML_LOG_ERROR and signaled by returning false.

bool append_copy_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * dst = node->src[1];
    const RemoteBinding * src_b = find_tensor_binding(src);
    const RemoteBinding * dst_b = find_tensor_binding(dst);
    const std::size_t nbytes = ggml_nbytes(src);
    if (src_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src_b, 0, nbytes) ||
        !remote_range_is_valid(*dst_b, 0, nbytes)) {
        GGML_LOG_ERROR("pynq: CPY node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_COPY },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC, src_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_NBYTES, nbytes },
        { P::F_SRC_OFFSET, src_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower COPY node=%s src=%s/%llu dst=%s/%llu bytes=%zu\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_b->handle),
            tensor_name(dst),
            static_cast<unsigned long long>(dst_b->handle),
            nbytes);
    }
    return true;
}

bool append_f32_binary_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    F32BinaryLowering lowering;
    get_f32_binary_lowering(node, &lowering);  // matches() already verified

    const RemoteBinding * src0_b = find_tensor_binding(lowering.src0);
    const RemoteBinding * src1_b = find_tensor_binding(lowering.src1);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    const std::size_t src0_nbytes = ggml_nbytes(lowering.src0);
    const std::size_t src1_nbytes = ggml_nbytes(lowering.src1);
    const std::size_t dst_nbytes = ggml_nbytes(node);
    if (src0_b == nullptr || src1_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src0_b, 0, src0_nbytes) ||
        !remote_range_is_valid(*src1_b, 0, src1_nbytes) ||
        !remote_range_is_valid(*dst_b, 0, dst_nbytes)) {
        GGML_LOG_ERROR("pynq: binary node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    const char * op_name = node->op == GGML_OP_ADD ? P::GOP_ADD_F32 : P::GOP_MUL_F32;
    ops->push_back({
        { P::F_OP, op_name },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC0, src0_b->handle },
        { P::F_SRC1, src1_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ROWS, node->ne[0] },
        { P::F_COLS, flattened_cols(node) },
        { P::F_SRC1_BROADCAST, lowering.src1_broadcast },
        { P::F_SRC0_OFFSET, src0_b->remote_offset },
        { P::F_SRC1_OFFSET, src1_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower %s node=%s src0=%s/%llu src1=%s/%llu "
            "dst=%llu rows=%lld cols=%lld broadcast=%s\n",
            op_name,
            tensor_name(node),
            tensor_name(lowering.src0),
            static_cast<unsigned long long>(src0_b->handle),
            tensor_name(lowering.src1),
            static_cast<unsigned long long>(src1_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(node->ne[0]),
            static_cast<long long>(flattened_cols(node)),
            lowering.src1_broadcast ? "true" : "false");
    }
    return true;
}

bool append_scale_f32_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_b = find_tensor_binding(src);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src_b, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: SCALE node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_SCALE_F32 },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC, src_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ELEMENTS, ggml_nelements(node) },
        { P::F_SCALE, op_param_f32(node, 0) },
        { P::F_BIAS, op_param_f32(node, 1) },
        { P::F_SRC_OFFSET, src_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SCALE_F32 node=%s src=%s/%llu dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

bool append_silu_f32_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_b = find_tensor_binding(src);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src_b, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: SILU node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_SILU_F32 },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC, src_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ELEMENTS, ggml_nelements(node) },
        { P::F_SRC_OFFSET, src_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SILU_F32 node=%s src=%s/%llu dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

bool append_swiglu_f32_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const RemoteBinding * src0_b = find_tensor_binding(src0);
    const RemoteBinding * src1_b = find_tensor_binding(src1);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    const std::size_t nbytes = ggml_nbytes(node);
    if (src0_b == nullptr || src1_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src0_b, 0, ggml_nbytes(src0)) ||
        !remote_range_is_valid(*src1_b, 0, ggml_nbytes(src1)) ||
        !remote_range_is_valid(*dst_b, 0, nbytes)) {
        GGML_LOG_ERROR("pynq: SWIGLU node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_SWIGLU_F32 },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC0, src0_b->handle },
        { P::F_SRC1, src1_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ELEMENTS, ggml_nelements(node) },
        { P::F_SRC0_OFFSET, src0_b->remote_offset },
        { P::F_SRC1_OFFSET, src1_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SWIGLU_F32 node=%s src0=%s/%llu src1=%s/%llu "
            "dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src0),
            static_cast<unsigned long long>(src0_b->handle),
            tensor_name(src1),
            static_cast<unsigned long long>(src1_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

bool append_rms_norm_f32_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_b = find_tensor_binding(src);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src_b, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: RMS_NORM node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_RMS_NORM_F32 },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC, src_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ROWS, node->ne[0] },
        { P::F_COLS, flattened_cols(node) },
        { P::F_EPS, op_param_f32(node, 0) },
        { P::F_SRC_OFFSET, src_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower RMS_NORM_F32 node=%s src=%s/%llu dst=%llu "
            "rows=%lld cols=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(node->ne[0]),
            static_cast<long long>(flattened_cols(node)));
    }
    return true;
}

bool append_rope_f32_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * pos = node->src[1];
    const RemoteBinding * src_b = find_tensor_binding(src);
    const RemoteBinding * pos_b = find_tensor_binding(pos);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src_b == nullptr || pos_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src_b, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*pos_b, 0, ggml_nbytes(pos)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: ROPE node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    // See supports_rope_f32 for the param layout. Indices are off-by-one
    // from "obvious" because ggml has legacy n_past at [0] and n_ctx at [3].
    const int32_t n_dims      = op_param_i32(node, 1);
    const int32_t mode        = op_param_i32(node, 2);
    const int32_t n_ctx_orig  = op_param_i32(node, 4);
    const float   freq_base   = op_param_f32(node, 5);
    const float   freq_scale  = op_param_f32(node, 6);
    const float   ext_factor  = op_param_f32(node, 7);
    const float   attn_factor = op_param_f32(node, 8);
    const float   beta_fast   = op_param_f32(node, 9);
    const float   beta_slow   = op_param_f32(node, 10);
    ops->push_back({
        { P::F_OP, P::GOP_ROPE_F32 },
        { P::F_NAME, tensor_name(node) },
        { P::F_SRC, src_b->handle },
        { P::F_POSITIONS, pos_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_HEAD_DIM, src->ne[0] },
        { P::F_N_HEAD, src->ne[1] },
        { P::F_N_TOKEN, src->ne[2] },
        { P::F_N_DIMS, n_dims },
        { P::F_MODE, mode },
        { P::F_N_CTX_ORIG, n_ctx_orig },
        { P::F_FREQ_BASE, freq_base },
        { P::F_FREQ_SCALE, freq_scale },
        { P::F_EXT_FACTOR, ext_factor },
        { P::F_ATTN_FACTOR, attn_factor },
        { P::F_BETA_FAST, beta_fast },
        { P::F_BETA_SLOW, beta_slow },
        { P::F_SRC_OFFSET, src_b->remote_offset },
        { P::F_POSITIONS_OFFSET, pos_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower ROPE_F32 node=%s src=%s/%llu pos=%s/%llu dst=%llu "
            "head_dim=%lld n_head=%lld n_token=%lld n_dims=%d mode=%d "
            "freq_base=%g freq_scale=%g\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_b->handle),
            tensor_name(pos),
            static_cast<unsigned long long>(pos_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(src->ne[0]),
            static_cast<long long>(src->ne[1]),
            static_cast<long long>(src->ne[2]),
            n_dims, mode, freq_base, freq_scale);
    }
    return true;
}

bool append_matmul_q1a8_op(const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * weights = node->src[0];
    const ggml_tensor * acts = node->src[1];
    const RemoteBinding * weights_b = find_tensor_binding(weights);
    const RemoteBinding * acts_b = find_tensor_binding(acts);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (weights_b == nullptr || acts_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*weights_b, 0, ggml_nbytes(weights)) ||
        !remote_range_is_valid(*acts_b, 0, ggml_nbytes(acts)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: MUL_MAT node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }
    ops->push_back({
        { P::F_OP, P::GOP_MATMUL_Q1A8 },
        { P::F_NAME, tensor_name(node) },
        { P::F_WEIGHTS, weights_b->handle },
        { P::F_ACTS, acts_b->handle },
        { P::F_DST, dst_b->handle },
        { P::F_ROWS, weights->ne[1] },
        { P::F_COLS, acts->ne[1] },
        { P::F_K, weights->ne[0] },
        { P::F_WEIGHTS_OFFSET, weights_b->remote_offset },
        { P::F_ACTS_OFFSET, acts_b->remote_offset },
        { P::F_DST_OFFSET, dst_b->remote_offset },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower MATMUL_Q1A8 node=%s weights=%s/%llu "
            "acts=%s/%llu dst=%llu rows=%lld cols=%lld k=%lld\n",
            tensor_name(node),
            tensor_name(weights),
            static_cast<unsigned long long>(weights_b->handle),
            tensor_name(acts),
            static_cast<unsigned long long>(acts_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(weights->ne[1]),
            static_cast<long long>(acts->ne[1]),
            static_cast<long long>(weights->ne[0]));
    }
    return true;
}

const char * type_tag(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_I32:  return "i32";
        case GGML_TYPE_I64:  return "i64";
        case GGML_TYPE_Q1_0: return "q1_0";
        default:             return "unknown";
    }
}

bool append_flash_attn_ext_f32_op(
    const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * mask = (GGML_MAX_SRC >= 4) ? node->src[3] : nullptr;
    const RemoteBinding * q_b = find_tensor_binding(q);
    const RemoteBinding * k_b = find_tensor_binding(k);
    const RemoteBinding * v_b = find_tensor_binding(v);
    const RemoteBinding * mask_b = mask ? find_tensor_binding(mask) : nullptr;
    const RemoteBinding * dst_b = find_tensor_binding(node);

    if (q_b == nullptr || k_b == nullptr || v_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*q_b, 0, ggml_nbytes(q)) ||
        !remote_range_is_valid(*k_b, 0, ggml_nbytes(k)) ||
        !remote_range_is_valid(*v_b, 0, ggml_nbytes(v)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: FLASH_ATTN_EXT node %s missing handles\n", node->name);
        return false;
    }
    if (mask != nullptr && (mask_b == nullptr ||
        !remote_range_is_valid(*mask_b, 0, ggml_nbytes(mask)))) {
        GGML_LOG_ERROR("pynq: FLASH_ATTN_EXT node %s mask handle invalid\n", node->name);
        return false;
    }

    float scale         = op_param_f32(node, 0);
    float max_bias      = op_param_f32(node, 1);
    float logit_softcap = op_param_f32(node, 2);

    nlohmann::json op = {
        { P::F_OP,             P::GOP_FLASH_ATTN_EXT_F32 },
        { P::F_NAME,           tensor_name(node) },
        { P::F_SRC0,           q_b->handle },
        { P::F_SRC0_OFFSET,    q_b->remote_offset },
        { P::F_K_TENSOR,       k_b->handle },
        { P::F_K_OFFSET,       k_b->remote_offset },
        { P::F_V_TENSOR,       v_b->handle },
        { P::F_V_OFFSET,       v_b->remote_offset },
        { P::F_DST,            dst_b->handle },
        { P::F_DST_OFFSET,     dst_b->remote_offset },
        { P::F_HAS_MASK,       mask != nullptr },
        { P::F_HEAD_DIM_Q,     q->ne[0] },
        { P::F_HEAD_DIM_V,     v->ne[0] },
        { P::F_N_TOKEN,        q->ne[1] },
        { P::F_N_HEAD,         q->ne[2] },
        { P::F_N_KV,           k->ne[1] },
        { P::F_N_HEAD_KV,      k->ne[2] },
        { P::F_SCALE,          scale },
        { P::F_MAX_BIAS,       max_bias },
        { P::F_LOGIT_SOFTCAP,  logit_softcap },
        { P::F_Q_NB1,          q->nb[1] },
        { P::F_Q_NB2,          q->nb[2] },
        { P::F_K_NB1,          k->nb[1] },
        { P::F_K_NB2,          k->nb[2] },
        { P::F_V_NB1,          v->nb[1] },
        { P::F_V_NB2,          v->nb[2] },
        { P::F_DST_NB1,        node->nb[1] },
        { P::F_DST_NB2,        node->nb[2] },
    };
    if (mask != nullptr) {
        op[P::F_MASK]        = mask_b->handle;
        op[P::F_MASK_OFFSET] = mask_b->remote_offset;
        op[P::F_MASK_NB1]    = mask->nb[1];
    }
    ops->push_back(std::move(op));
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower FLASH_ATTN_EXT_F32 node=%s q=%s/%llu k=%s/%llu "
            "v=%s/%llu dst=%llu head_dim_q=%lld head_dim_v=%lld "
            "n_head=%lld n_head_kv=%lld n_kv=%lld n_token=%lld scale=%g mask=%s\n",
            tensor_name(node),
            tensor_name(q), static_cast<unsigned long long>(q_b->handle),
            tensor_name(k), static_cast<unsigned long long>(k_b->handle),
            tensor_name(v), static_cast<unsigned long long>(v_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(q->ne[0]),
            static_cast<long long>(v->ne[0]),
            static_cast<long long>(q->ne[2]),
            static_cast<long long>(k->ne[2]),
            static_cast<long long>(k->ne[1]),
            static_cast<long long>(q->ne[1]),
            scale,
            mask ? "yes" : "no");
    }
    return true;
}

bool append_get_rows_op(
    const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const RemoteBinding * src0_b = find_tensor_binding(src0);
    const RemoteBinding * src1_b = find_tensor_binding(src1);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src0_b == nullptr || src1_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src0_b, 0, ggml_nbytes(src0)) ||
        !remote_range_is_valid(*src1_b, 0, ggml_nbytes(src1)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: GET_ROWS node %s missing handles\n", node->name);
        return false;
    }

    const int64_t n_indices = src1->ne[0] * src1->ne[1] * src1->ne[2];
    ops->push_back({
        { P::F_OP,             P::GOP_GET_ROWS },
        { P::F_NAME,           tensor_name(node) },
        { P::F_SRC0,           src0_b->handle },
        { P::F_SRC0_OFFSET,    src0_b->remote_offset },
        { P::F_INDICES,        src1_b->handle },
        { P::F_INDICES_OFFSET, src1_b->remote_offset },
        { P::F_DST,            dst_b->handle },
        { P::F_DST_OFFSET,     dst_b->remote_offset },
        { P::F_SRC0_TYPE,      type_tag(src0->type) },
        { P::F_INDICES_TYPE,   type_tag(src1->type) },
        { P::F_HEAD_DIM,       src0->ne[0] },     // row width (n_embd)
        { P::F_NE01,           src0->ne[1] },     // total rows in src0
        { P::F_NE02,           src0->ne[2] },
        { P::F_NE03,           src0->ne[3] },
        { P::F_N_INDICES,      n_indices },
        { P::F_NE10,           src1->ne[0] },
        { P::F_NE11,           src1->ne[1] },
        { P::F_NE12,           src1->ne[2] },
        { P::F_SRC0_NB1,       src0->nb[1] },
        { P::F_SRC0_NB2,       src0->nb[2] },
        { P::F_SRC0_NB3,       src0->nb[3] },
        { P::F_INDICES_NB1,    src1->nb[1] },
        { P::F_INDICES_NB2,    src1->nb[2] },
        { P::F_DST_NB1,        node->nb[1] },
        { P::F_DST_NB2,        node->nb[2] },
        { P::F_DST_NB3,        node->nb[3] },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower GET_ROWS node=%s src0=%s/%llu type=%s "
            "indices=%s/%llu n_indices=%lld head_dim=%lld\n",
            tensor_name(node),
            tensor_name(src0),
            static_cast<unsigned long long>(src0_b->handle),
            type_tag(src0->type),
            tensor_name(src1),
            static_cast<unsigned long long>(src1_b->handle),
            static_cast<long long>(n_indices),
            static_cast<long long>(src0->ne[0]));
    }
    return true;
}

bool append_set_rows_op(
    const ggml_tensor * node, nlohmann::json * ops, nlohmann::json * outputs) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const RemoteBinding * src0_b = find_tensor_binding(src0);
    const RemoteBinding * src1_b = find_tensor_binding(src1);
    const RemoteBinding * dst_b = find_tensor_binding(node);
    if (src0_b == nullptr || src1_b == nullptr || dst_b == nullptr ||
        !remote_range_is_valid(*src0_b, 0, ggml_nbytes(src0)) ||
        !remote_range_is_valid(*src1_b, 0, ggml_nbytes(src1)) ||
        !remote_range_is_valid(*dst_b, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: SET_ROWS node %s missing handles\n", node->name);
        return false;
    }

    ops->push_back({
        { P::F_OP,             P::GOP_SET_ROWS },
        { P::F_NAME,           tensor_name(node) },
        { P::F_SRC0,           src0_b->handle },
        { P::F_SRC0_OFFSET,    src0_b->remote_offset },
        { P::F_INDICES,        src1_b->handle },
        { P::F_INDICES_OFFSET, src1_b->remote_offset },
        { P::F_DST,            dst_b->handle },
        { P::F_DST_OFFSET,     dst_b->remote_offset },
        { P::F_INDICES_TYPE,   type_tag(src1->type) },
        { P::F_DST_TYPE,       type_tag(node->type) },  // f16 for KV cache
        { P::F_HEAD_DIM,       src0->ne[0] },           // nc (row width)
        { P::F_NE01,           src0->ne[1] },           // n rows in src0
        { P::F_NE02,           src0->ne[2] },
        { P::F_NE03,           src0->ne[3] },
        { P::F_NE10,           src1->ne[0] },
        { P::F_NE11,           src1->ne[1] },
        { P::F_NE12,           src1->ne[2] },
        { P::F_SRC0_NB1,       src0->nb[1] },
        { P::F_SRC0_NB2,       src0->nb[2] },
        { P::F_SRC0_NB3,       src0->nb[3] },
        { P::F_INDICES_NB1,    src1->nb[1] },
        { P::F_INDICES_NB2,    src1->nb[2] },
        { P::F_DST_NB1,        node->nb[1] },
        { P::F_DST_NB2,        node->nb[2] },
        { P::F_DST_NB3,        node->nb[3] },
    });
    outputs->push_back(dst_b->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SET_ROWS node=%s src0=%s/%llu indices=%s/%llu "
            "dst=%llu head_dim=%lld n_rows=%lld dst_type=%s indices_type=%s\n",
            tensor_name(node),
            tensor_name(src0),
            static_cast<unsigned long long>(src0_b->handle),
            tensor_name(src1),
            static_cast<unsigned long long>(src1_b->handle),
            static_cast<unsigned long long>(dst_b->handle),
            static_cast<long long>(src0->ne[0]),
            static_cast<long long>(src0->ne[1]),
            type_tag(node->type),
            type_tag(src1->type));
    }
    return true;
}

// -- op lowering table ----------------------------------------------------
//
// One entry per supported ggml op. ``matches`` decides whether this op
// participates; ``append`` emits its wire form. Adding a new op = one
// entry here (and one Kernel registration on the board).

struct OpLowering {
    bool (*matches)(const ggml_tensor *);
    bool (*append)(const ggml_tensor *, nlohmann::json *, nlohmann::json *);
};

constexpr OpLowering k_lowerings[] = {
    { supports_raw_copy,           append_copy_op },
    { supports_matmul_q1a8,        append_matmul_q1a8_op },
    { supports_f32_binary,         append_f32_binary_op },
    { supports_scale_f32,          append_scale_f32_op },
    { supports_silu_f32,           append_silu_f32_op },
    { supports_swiglu_f32,         append_swiglu_f32_op },
    { supports_rms_norm_f32,       append_rms_norm_f32_op },
    { supports_rope_f32,           append_rope_f32_op },
    { supports_flash_attn_ext_f32, append_flash_attn_ext_f32_op },
    { supports_get_rows,           append_get_rows_op },
    { supports_set_rows,           append_set_rows_op },
};

const OpLowering * lookup_lowering(const ggml_tensor * op) {
    // An op that produces zero output elements is a no-op. llama.cpp builds
    // these on every non-final prompt ubatch, where n_outputs==0: the empty
    // inp_out_ids GET_ROWS ([n_embd, 0]) and the lm_head matmul fed from it
    // ([vocab, 0]). The board's kernels require positive extents (ne10 > 0,
    // rows/cols > 0), so decline empty work and let the CPU backend run it.
    if (op != nullptr && ggml_nelements(op) == 0) {
        return nullptr;
    }
    for (const auto & entry : k_lowerings) {
        if (entry.matches(op)) {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace

bool device_supports_op_impl(const ggml_tensor * op) {
    if (op != nullptr && is_metadata_op(op->op)) {
        return true;
    }
    const bool ok = lookup_lowering(op) != nullptr;
    if (!ok) {
        // Record so dump_unsupported_op_census() at backend_free can show
        // which op kinds are forcing scheduler splits.
        record_unsupported_op(op);
    }
    return ok;
}

bool device_offload_op_impl(const ggml_tensor * op) {
    // Same predicates as device_supports_op, minus the metadata-op pass —
    // ggml's scheduler treats those as no-cost reshapes.
    return lookup_lowering(op) != nullptr;
}

enum ggml_status backend_graph_compute_impl(ggml_backend_t backend, ggml_cgraph * cgraph) {
    nlohmann::json ops = nlohmann::json::array();
    nlohmann::json outputs = nlohmann::json::array();

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (is_metadata_op(node->op)) {
            continue;
        }
        const OpLowering * lowering = lookup_lowering(node);
        if (lowering == nullptr) {
            GGML_LOG_ERROR("pynq: unsupported graph op %s in node %s\n",
                ggml_op_name(node->op),
                node->name);
            return GGML_STATUS_FAILED;
        }
        if (!lowering->append(node, &ops, &outputs)) {
            return GGML_STATUS_FAILED;
        }
    }

    if (ops.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    GGML_UNUSED(backend);
    try {
        const std::size_t graph_call = trace_counters().graph_calls.fetch_add(1) + 1;
        if (trace_enabled()) {
            tracef(
                "pynq trace: RUN_GRAPH #%zu submit ops=%zu outputs=%zu\n",
                graph_call,
                ops.size(),
                outputs.size());
        }
        const pynq::RpcResponse response = shared_client().call(
            P::OP_RUN_GRAPH,
            {
                { P::F_GRAPH_VERSION, P::GRAPH_VERSION },
                { P::F_OPS, ops },
                { P::F_OUTPUTS, outputs },
            });
        if (trace_enabled()) {
            tracef(
                "pynq trace: RUN_GRAPH #%zu complete result=%s\n",
                graph_call,
                response.result.dump().c_str());
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("pynq: RUN_GRAPH failed: %s\n", exc.what());
        return GGML_STATUS_FAILED;
    }
}

} // namespace pynq
