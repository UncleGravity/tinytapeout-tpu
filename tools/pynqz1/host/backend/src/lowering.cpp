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
    { supports_raw_copy,      append_copy_op },
    { supports_matmul_q1a8,   append_matmul_q1a8_op },
    { supports_f32_binary,    append_f32_binary_op },
    { supports_scale_f32,     append_scale_f32_op },
    { supports_silu_f32,      append_silu_f32_op },
    { supports_swiglu_f32,    append_swiglu_f32_op },
    { supports_rms_norm_f32,  append_rms_norm_f32_op },
};

const OpLowering * lookup_lowering(const ggml_tensor * op) {
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
