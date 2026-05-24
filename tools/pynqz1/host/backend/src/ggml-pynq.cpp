#include "ggml-pynq.h"

#include "client.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

namespace {

constexpr const char * k_backend_name = "PYNQ";
constexpr const char * k_device_description = "PYNQ-Z1 bonsaid remote tensor backend";
constexpr size_t k_alignment = 64;
constexpr size_t k_fill_chunk_size = 1 * 1024 * 1024;
constexpr int k_runtime_abi_version = 1;
constexpr int k_graph_version = 1;

static std::atomic<size_t> g_trace_allocated_bytes { 0 };
static std::atomic<size_t> g_trace_uploaded_bytes { 0 };
static std::atomic<size_t> g_trace_downloaded_bytes { 0 };
static std::atomic<size_t> g_trace_graph_calls { 0 };

struct BackendContext {
    pynq::Endpoint endpoint;
};

struct RemoteBinding {
    uint64_t handle = 0;
    size_t handle_nbytes = 0;
    size_t tensor_nbytes = 0;
    size_t remote_offset = 0;
};

struct F32BinaryLowering {
    const ggml_tensor * src0 = nullptr;
    const ggml_tensor * src1 = nullptr;
    bool src1_broadcast = false;
};

struct BufferContext {
    explicit BufferContext(pynq::Endpoint endpoint) : rpc(endpoint) {
    }

    pynq::RpcClient rpc;
    void * base = nullptr;
    size_t size = 0;
    uint64_t remote_handle = 0;
    size_t remote_nbytes = 0;
    std::unordered_map<const ggml_tensor *, RemoteBinding> bindings;
};

static ggml_guid_t backend_guid() {
    static ggml_guid guid = {
        0x70, 0x79, 0x6e, 0x71, 0x2d, 0x62, 0x6f, 0x6e,
        0x73, 0x61, 0x69, 0x2d, 0x7a, 0x31, 0x30, 0x31,
    };
    return &guid;
}

static size_t add_checked(size_t lhs, size_t rhs) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        throw pynq::RpcError("tensor range overflows size_t");
    }
    return lhs + rhs;
}

static bool remote_range_is_valid(const RemoteBinding & binding, size_t offset, size_t size) {
    try {
        const size_t tensor_end = add_checked(offset, size);
        const size_t remote_end = add_checked(binding.remote_offset, tensor_end);
        return tensor_end <= binding.tensor_nbytes && remote_end <= binding.handle_nbytes;
    } catch (const pynq::RpcError &) {
        return false;
    }
}

static BufferContext * buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<BufferContext *>(buffer->context);
}

static const RemoteBinding * find_binding(BufferContext * ctx, const ggml_tensor * tensor) {
    const auto it = ctx->bindings.find(tensor);
    return it == ctx->bindings.end() ? nullptr : &it->second;
}

static bool trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("PYNQ_TRACE");
        return value != nullptr &&
            value[0] != '\0' &&
            std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static void tracef(const char * format, ...) {
    if (!trace_enabled()) {
        return;
    }

    va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fflush(stderr);
}

static const char * tensor_name(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->name[0] != '\0' ? tensor->name : "(unnamed)";
}

static double mib(size_t nbytes) {
    return static_cast<double>(nbytes) / (1024.0 * 1024.0);
}

static void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc) {
    GGML_LOG_ERROR("pynq: %s failed for tensor %s: %s\n",
        action,
        tensor_name(tensor),
        exc.what());
}

static void upload_fill(BufferContext * ctx, uint64_t handle, size_t offset, size_t size, uint8_t value) {
    std::vector<uint8_t> fill(std::min(size, k_fill_chunk_size), value);
    size_t done = 0;
    while (done != size) {
        const size_t chunk = std::min(size - done, fill.size());
        ctx->rpc.call(
            "UPLOAD_TENSOR",
            {
                { "handle", handle },
                { "offset", add_checked(offset, done) },
            },
            fill.data(),
            chunk);
        done += chunk;
    }
}

static bool tensor_buffer_offset(
    const BufferContext * ctx,
    const ggml_tensor * tensor,
    size_t * offset) {
    if (ctx->size == 0) {
        *offset = 0;
        return ggml_nbytes(tensor) == 0;
    }
    if (ctx->base == nullptr || tensor->data == nullptr) {
        return false;
    }

    const uintptr_t base = reinterpret_cast<uintptr_t>(ctx->base);
    const uintptr_t data = reinterpret_cast<uintptr_t>(tensor->data);
    if (data < base) {
        return false;
    }

    const size_t local_offset = static_cast<size_t>(data - base);
    const size_t nbytes = ggml_nbytes(tensor);
    try {
        if (add_checked(local_offset, nbytes) > ctx->size) {
            return false;
        }
    } catch (const pynq::RpcError &) {
        return false;
    }

    *offset = local_offset;
    return true;
}

static void free_buffer_allocation(BufferContext * ctx) {
    if (ctx->remote_handle != 0) {
        try {
            ctx->rpc.call("FREE_TENSOR", { { "handle", ctx->remote_handle } });
            if (trace_enabled()) {
                tracef(
                    "pynq trace: free handle=%llu bytes=%zu\n",
                    static_cast<unsigned long long>(ctx->remote_handle),
                    ctx->remote_nbytes);
            }
        } catch (const std::exception & exc) {
            GGML_LOG_WARN("pynq: FREE_TENSOR failed for handle %llu: %s\n",
                static_cast<unsigned long long>(ctx->remote_handle),
                exc.what());
        }
        ctx->remote_handle = 0;
        ctx->remote_nbytes = 0;
    }
    ctx->bindings.clear();
}

static void * reserve_fake_base(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    void * base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return base == MAP_FAILED ? nullptr : base;
}

static void release_fake_base(void * base, size_t size) {
    if (base != nullptr && size != 0) {
        munmap(base, size);
    }
}

static void pynq_buffer_free(ggml_backend_buffer_t buffer) {
    BufferContext * ctx = buffer_ctx(buffer);
    free_buffer_allocation(ctx);
    release_fake_base(ctx->base, ctx->size);
    delete ctx;
}

static void * pynq_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer_ctx(buffer)->base;
}

static bool is_pynq_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.get_base == pynq_buffer_get_base;
}

static const RemoteBinding * find_tensor_binding(const ggml_tensor * tensor) {
    if (tensor == nullptr || !is_pynq_buffer(tensor->buffer)) {
        return nullptr;
    }
    return find_binding(buffer_ctx(tensor->buffer), tensor);
}

static enum ggml_status pynq_buffer_init_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor) {
    BufferContext * ctx = buffer_ctx(buffer);
    try {
        if (tensor->view_src != nullptr) {
            const RemoteBinding * src = find_binding(ctx, tensor->view_src);
            if (src == nullptr) {
                GGML_LOG_ERROR("pynq: view source for tensor %s is not bound\n", tensor->name);
                return GGML_STATUS_FAILED;
            }

            RemoteBinding binding = *src;
            binding.remote_offset = add_checked(binding.remote_offset, tensor->view_offs);
            binding.tensor_nbytes = ggml_nbytes(tensor);
            if (!remote_range_is_valid(binding, 0, binding.tensor_nbytes)) {
                GGML_LOG_ERROR("pynq: view tensor %s exceeds source allocation\n", tensor->name);
                return GGML_STATUS_FAILED;
            }
            ctx->bindings.emplace(tensor, binding);
            if (trace_enabled()) {
                tracef(
                    "pynq trace: view name=%s type=%s bytes=%zu handle=%llu offset=%zu\n",
                    tensor_name(tensor),
                    ggml_type_name(tensor->type),
                    binding.tensor_nbytes,
                    static_cast<unsigned long long>(binding.handle),
                    binding.remote_offset);
            }
            return GGML_STATUS_SUCCESS;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        size_t remote_offset = 0;
        if (!tensor_buffer_offset(ctx, tensor, &remote_offset)) {
            GGML_LOG_ERROR("pynq: tensor %s is outside its PYNQ buffer arena\n", tensor->name);
            return GGML_STATUS_FAILED;
        }
        if (nbytes != 0 && ctx->remote_handle == 0) {
            GGML_LOG_ERROR("pynq: tensor %s has no backing PYNQ buffer allocation\n", tensor->name);
            return GGML_STATUS_FAILED;
        }

        RemoteBinding binding {
            ctx->remote_handle,
            ctx->remote_nbytes,
            nbytes,
            remote_offset,
        };
        if (!remote_range_is_valid(binding, 0, nbytes)) {
            GGML_LOG_ERROR("pynq: tensor %s exceeds its PYNQ buffer allocation\n", tensor->name);
            return GGML_STATUS_FAILED;
        }
        ctx->bindings.emplace(tensor, binding);
        if (trace_enabled()) {
            tracef(
                "pynq trace: bind name=%s type=%s shape=[%lld,%lld,%lld,%lld] "
                "bytes=%zu handle=%llu offset=%zu buffer_bytes=%zu\n",
                tensor_name(tensor),
                ggml_type_name(tensor->type),
                static_cast<long long>(tensor->ne[0]),
                static_cast<long long>(tensor->ne[1]),
                static_cast<long long>(tensor->ne[2]),
                static_cast<long long>(tensor->ne[3]),
                nbytes,
                static_cast<unsigned long long>(ctx->remote_handle),
                remote_offset,
                ctx->remote_nbytes);
        }
        return GGML_STATUS_SUCCESS;
    } catch (const std::exception & exc) {
        log_rpc_failure("tensor init", tensor, exc);
        return GGML_STATUS_FAILED;
    }
}

static void pynq_buffer_memset_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    uint8_t value,
    size_t offset,
    size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid memset range for tensor %s\n", tensor->name);
        return;
    }

    try {
        upload_fill(ctx, binding->handle, add_checked(binding->remote_offset, offset), size, value);
        if (trace_enabled()) {
            const size_t uploaded_total =
                g_trace_uploaded_bytes.fetch_add(size) + size;
            tracef(
                "pynq trace: memset name=%s handle=%llu offset=%zu bytes=%zu "
                "value=%u uploaded_total=%.2f MiB\n",
                tensor_name(tensor),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                size,
                static_cast<unsigned>(value),
                mib(uploaded_total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("memset", tensor, exc);
    }
}

static void pynq_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    const void * data,
    size_t offset,
    size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid upload range for tensor %s\n", tensor->name);
        return;
    }

    try {
        ctx->rpc.call(
            "UPLOAD_TENSOR",
            {
                { "handle", binding->handle },
                { "offset", add_checked(binding->remote_offset, offset) },
            },
            data,
            size);
        if (trace_enabled()) {
            const size_t uploaded_total =
                g_trace_uploaded_bytes.fetch_add(size) + size;
            tracef(
                "pynq trace: upload name=%s type=%s handle=%llu offset=%zu "
                "bytes=%zu uploaded_total=%.2f MiB\n",
                tensor_name(tensor),
                ggml_type_name(tensor->type),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                size,
                mib(uploaded_total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("upload", tensor, exc);
    }
}

static void pynq_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const ggml_tensor * tensor,
    void * data,
    size_t offset,
    size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid download range for tensor %s\n", tensor->name);
        return;
    }

    try {
        const pynq::RpcResponse response = ctx->rpc.call(
            "DOWNLOAD_TENSOR",
            {
                { "handle", binding->handle },
                { "offset", add_checked(binding->remote_offset, offset) },
                { "size", size },
            });
        if (response.payload.size() != size) {
            throw pynq::RpcError("DOWNLOAD_TENSOR returned an unexpected payload size");
        }
        std::memcpy(data, response.payload.data(), size);
        if (trace_enabled()) {
            const size_t downloaded_total =
                g_trace_downloaded_bytes.fetch_add(size) + size;
            tracef(
                "pynq trace: download name=%s type=%s handle=%llu offset=%zu "
                "bytes=%zu downloaded_total=%.2f MiB\n",
                tensor_name(tensor),
                ggml_type_name(tensor->type),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                size,
                mib(downloaded_total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("download", tensor, exc);
    }
}

static bool pynq_buffer_cpy_tensor(
    ggml_backend_buffer_t buffer,
    const ggml_tensor * src,
    ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void pynq_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    BufferContext * ctx = buffer_ctx(buffer);
    if (ctx->remote_handle == 0 || ctx->remote_nbytes == 0) {
        return;
    }

    try {
        upload_fill(ctx, ctx->remote_handle, 0, ctx->remote_nbytes, value);
        if (trace_enabled()) {
            const size_t uploaded_total =
                g_trace_uploaded_bytes.fetch_add(ctx->remote_nbytes) + ctx->remote_nbytes;
            tracef(
                "pynq trace: clear handle=%llu bytes=%zu value=%u uploaded_total=%.2f MiB\n",
                static_cast<unsigned long long>(ctx->remote_handle),
                ctx->remote_nbytes,
                static_cast<unsigned>(value),
                mib(uploaded_total));
        }
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("pynq: clear failed for handle %llu: %s\n",
            static_cast<unsigned long long>(ctx->remote_handle),
            exc.what());
    }
}

static void pynq_buffer_reset(ggml_backend_buffer_t buffer) {
    buffer_ctx(buffer)->bindings.clear();
}

static const ggml_backend_buffer_i buffer_i = {
    /* .free_buffer   = */ pynq_buffer_free,
    /* .get_base      = */ pynq_buffer_get_base,
    /* .init_tensor   = */ pynq_buffer_init_tensor,
    /* .memset_tensor = */ pynq_buffer_memset_tensor,
    /* .set_tensor    = */ pynq_buffer_set_tensor,
    /* .get_tensor    = */ pynq_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ pynq_buffer_cpy_tensor,
    /* .clear         = */ pynq_buffer_clear,
    /* .reset         = */ pynq_buffer_reset,
};

static const char * buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "PYNQ_REMOTE";
}

static ggml_backend_buffer_t buffer_type_alloc_buffer(
    ggml_backend_buffer_type_t buft,
    size_t size) {
    std::unique_ptr<BufferContext> ctx;
    try {
        ctx.reset(new BufferContext(pynq::endpoint_from_env()));
        ctx->base = reserve_fake_base(size);
        ctx->size = size;
        if (size != 0 && ctx->base == nullptr) {
            GGML_LOG_ERROR("pynq: cannot reserve %zu bytes of local tensor address space\n", size);
            return nullptr;
        }

        if (size != 0) {
            const pynq::RpcResponse response = ctx->rpc.call(
                "ALLOC_TENSOR",
                {
                    { "nbytes", size },
                    { "shape", nlohmann::json::array({ size }) },
                    { "dtype", "u8" },
                    { "usage", "ggml-buffer" },
                    { "layout", "ggml-buffer" },
                    { "alignment", k_alignment },
                });
            const nlohmann::json & remote = response.result.at("tensor");
            ctx->remote_handle = remote.at("handle").get<uint64_t>();
            ctx->remote_nbytes = remote.at("nbytes").get<size_t>();
        }

        if (trace_enabled()) {
            const size_t allocated_total =
                g_trace_allocated_bytes.fetch_add(ctx->remote_nbytes) + ctx->remote_nbytes;
            tracef(
                "pynq trace: buffer reserve bytes=%zu fake_base=%p handle=%llu "
                "remote_bytes=%zu allocated_total=%.2f MiB\n",
                size,
                ctx->base,
                static_cast<unsigned long long>(ctx->remote_handle),
                ctx->remote_nbytes,
                mib(allocated_total));
        }
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft, buffer_i, ctx.get(), size);
        if (buffer == nullptr) {
            free_buffer_allocation(ctx.get());
            release_fake_base(ctx->base, ctx->size);
            return nullptr;
        }
        ctx.release();
        return buffer;
    } catch (const std::exception & exc) {
        if (ctx != nullptr) {
            free_buffer_allocation(ctx.get());
            release_fake_base(ctx->base, ctx->size);
        }
        GGML_LOG_ERROR("pynq: buffer allocation failed before tensor init: %s\n", exc.what());
        return nullptr;
    }
}

static size_t buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return k_alignment;
}

static size_t memory_value(const nlohmann::json & result, const char * key) {
    return result.at("memory").at(key).get<size_t>();
}

static size_t buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    try {
        const pynq::Endpoint endpoint = pynq::endpoint_from_env();
        tracef(
            "pynq trace: buffer max_size query endpoint=%s:%u\n",
            endpoint.host.c_str(),
            static_cast<unsigned>(endpoint.port));
        pynq::RpcClient rpc(endpoint);
        const pynq::RpcResponse response = rpc.call("MEMORY");
        const size_t free_bytes = memory_value(response.result, "free_bytes");
        tracef("pynq trace: buffer max_size free=%.2f MiB\n", mib(free_bytes));
        return free_bytes;
    } catch (const std::exception & exc) {
        tracef("pynq trace: buffer max_size query failed: %s\n", exc.what());
        return std::numeric_limits<size_t>::max();
    }
}

static size_t buffer_type_get_alloc_size(
    ggml_backend_buffer_type_t buft,
    const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i buffer_type_i = {
    /* .get_name       = */ buffer_type_get_name,
    /* .alloc_buffer   = */ buffer_type_alloc_buffer,
    /* .get_alignment  = */ buffer_type_get_alignment,
    /* .get_max_size   = */ buffer_type_get_max_size,
    /* .get_alloc_size = */ buffer_type_get_alloc_size,
    /* .is_host        = */ buffer_type_is_host,
};

static const char * backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return k_backend_name;
}

static void backend_free(ggml_backend_t backend) {
    delete static_cast<BackendContext *>(backend->context);
    delete backend;
}

static bool is_metadata_op(ggml_op op) {
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

static bool supports_raw_copy(const ggml_tensor * op) {
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

static bool same_shape(const ggml_tensor * lhs, const ggml_tensor * rhs) {
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

static bool is_contiguous_f32(const ggml_tensor * tensor) {
    return tensor != nullptr &&
        tensor->type == GGML_TYPE_F32 &&
        tensor->ne[0] > 0 &&
        ggml_nelements(tensor) > 0 &&
        ggml_is_contiguous(tensor);
}

static int64_t flattened_cols(const ggml_tensor * tensor) {
    return ggml_nelements(tensor) / tensor->ne[0];
}

static float op_param_f32(const ggml_tensor * tensor, int index) {
    float value = 0.0f;
    std::memcpy(
        &value,
        reinterpret_cast<const char *>(tensor->op_params) + index * sizeof(value),
        sizeof(value));
    return value;
}

static bool is_row_broadcast_for(const ggml_tensor * src, const ggml_tensor * dst) {
    if (!is_contiguous_f32(src) ||
        !is_contiguous_f32(dst) ||
        src->ne[0] != dst->ne[0]) {
        return false;
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (src->ne[dim] != 1) {
            return false;
        }
    }
    return true;
}

static bool get_f32_binary_lowering(
    const ggml_tensor * op,
    F32BinaryLowering * lowering) {
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

static bool supports_f32_binary(const ggml_tensor * op) {
    F32BinaryLowering lowering;
    return get_f32_binary_lowering(op, &lowering);
}

static bool supports_scale_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_SCALE &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

static bool supports_silu_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_UNARY &&
        ggml_get_unary_op(op) == GGML_UNARY_OP_SILU &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

static bool supports_swiglu_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_GLU &&
        ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        is_contiguous_f32(op->src[1]) &&
        same_shape(op, op->src[0]) &&
        same_shape(op, op->src[1]);
}

static bool supports_rms_norm_f32(const ggml_tensor * op) {
    return op != nullptr &&
        op->op == GGML_OP_RMS_NORM &&
        is_contiguous_f32(op) &&
        is_contiguous_f32(op->src[0]) &&
        same_shape(op, op->src[0]);
}

static bool supports_matmul_q1a8(const ggml_tensor * op) {
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

    return ggml_is_contiguous(weights) &&
        ggml_is_contiguous(acts) &&
        ggml_is_contiguous(op);
}

static bool append_copy_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_raw_copy(node)) {
        GGML_LOG_ERROR("pynq: CPY node %s is not a contiguous same-type byte copy\n", node->name);
        return false;
    }

    const ggml_tensor * src = node->src[0];
    const ggml_tensor * dst = node->src[1];
    const RemoteBinding * src_binding = find_tensor_binding(src);
    const RemoteBinding * dst_binding = find_tensor_binding(dst);
    const size_t nbytes = ggml_nbytes(src);
    if (src_binding == nullptr || dst_binding == nullptr ||
        !remote_range_is_valid(*src_binding, 0, nbytes) ||
        !remote_range_is_valid(*dst_binding, 0, nbytes)) {
        GGML_LOG_ERROR("pynq: CPY node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "COPY" },
        { "name", tensor_name(node) },
        { "src", src_binding->handle },
        { "dst", dst_binding->handle },
        { "nbytes", nbytes },
        { "src_offset", src_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower COPY node=%s src=%s/%llu dst=%s/%llu bytes=%zu\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_binding->handle),
            tensor_name(dst),
            static_cast<unsigned long long>(dst_binding->handle),
            nbytes);
    }
    return true;
}

static bool append_f32_binary_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    F32BinaryLowering lowering;
    if (!get_f32_binary_lowering(node, &lowering)) {
        GGML_LOG_ERROR("pynq: binary node %s is not a supported F32 op\n", node->name);
        return false;
    }

    const RemoteBinding * src0_binding = find_tensor_binding(lowering.src0);
    const RemoteBinding * src1_binding = find_tensor_binding(lowering.src1);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    const size_t src0_nbytes = ggml_nbytes(lowering.src0);
    const size_t src1_nbytes = ggml_nbytes(lowering.src1);
    const size_t dst_nbytes = ggml_nbytes(node);
    if (src0_binding == nullptr ||
        src1_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*src0_binding, 0, src0_nbytes) ||
        !remote_range_is_valid(*src1_binding, 0, src1_nbytes) ||
        !remote_range_is_valid(*dst_binding, 0, dst_nbytes)) {
        GGML_LOG_ERROR("pynq: binary node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    const char * op_name = node->op == GGML_OP_ADD ? "ADD_F32" : "MUL_F32";
    ops->push_back({
        { "op", op_name },
        { "name", tensor_name(node) },
        { "src0", src0_binding->handle },
        { "src1", src1_binding->handle },
        { "dst", dst_binding->handle },
        { "rows", node->ne[0] },
        { "cols", flattened_cols(node) },
        { "src1_broadcast", lowering.src1_broadcast },
        { "src0_offset", src0_binding->remote_offset },
        { "src1_offset", src1_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower %s node=%s src0=%s/%llu src1=%s/%llu "
            "dst=%llu rows=%lld cols=%lld broadcast=%s\n",
            op_name,
            tensor_name(node),
            tensor_name(lowering.src0),
            static_cast<unsigned long long>(src0_binding->handle),
            tensor_name(lowering.src1),
            static_cast<unsigned long long>(src1_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(node->ne[0]),
            static_cast<long long>(flattened_cols(node)),
            lowering.src1_broadcast ? "true" : "false");
    }
    return true;
}

static bool append_scale_f32_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_scale_f32(node)) {
        GGML_LOG_ERROR("pynq: SCALE node %s is not a supported F32 scale\n", node->name);
        return false;
    }

    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_binding = find_tensor_binding(src);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    if (src_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*src_binding, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_binding, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: SCALE node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "SCALE_F32" },
        { "name", tensor_name(node) },
        { "src", src_binding->handle },
        { "dst", dst_binding->handle },
        { "elements", ggml_nelements(node) },
        { "scale", op_param_f32(node, 0) },
        { "bias", op_param_f32(node, 1) },
        { "src_offset", src_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SCALE_F32 node=%s src=%s/%llu dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

static bool append_silu_f32_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_silu_f32(node)) {
        GGML_LOG_ERROR("pynq: SILU node %s is not a supported F32 unary op\n", node->name);
        return false;
    }

    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_binding = find_tensor_binding(src);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    if (src_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*src_binding, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_binding, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: SILU node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "SILU_F32" },
        { "name", tensor_name(node) },
        { "src", src_binding->handle },
        { "dst", dst_binding->handle },
        { "elements", ggml_nelements(node) },
        { "src_offset", src_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SILU_F32 node=%s src=%s/%llu dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

static bool append_swiglu_f32_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_swiglu_f32(node)) {
        GGML_LOG_ERROR("pynq: SWIGLU node %s is not a supported F32 split GLU op\n", node->name);
        return false;
    }

    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const RemoteBinding * src0_binding = find_tensor_binding(src0);
    const RemoteBinding * src1_binding = find_tensor_binding(src1);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    const size_t nbytes = ggml_nbytes(node);
    if (src0_binding == nullptr ||
        src1_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*src0_binding, 0, ggml_nbytes(src0)) ||
        !remote_range_is_valid(*src1_binding, 0, ggml_nbytes(src1)) ||
        !remote_range_is_valid(*dst_binding, 0, nbytes)) {
        GGML_LOG_ERROR("pynq: SWIGLU node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "SWIGLU_F32" },
        { "name", tensor_name(node) },
        { "src0", src0_binding->handle },
        { "src1", src1_binding->handle },
        { "dst", dst_binding->handle },
        { "elements", ggml_nelements(node) },
        { "src0_offset", src0_binding->remote_offset },
        { "src1_offset", src1_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower SWIGLU_F32 node=%s src0=%s/%llu src1=%s/%llu "
            "dst=%llu elements=%lld\n",
            tensor_name(node),
            tensor_name(src0),
            static_cast<unsigned long long>(src0_binding->handle),
            tensor_name(src1),
            static_cast<unsigned long long>(src1_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(ggml_nelements(node)));
    }
    return true;
}

static bool append_rms_norm_f32_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_rms_norm_f32(node)) {
        GGML_LOG_ERROR("pynq: RMS_NORM node %s is not a supported F32 norm\n", node->name);
        return false;
    }

    const ggml_tensor * src = node->src[0];
    const RemoteBinding * src_binding = find_tensor_binding(src);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    if (src_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*src_binding, 0, ggml_nbytes(src)) ||
        !remote_range_is_valid(*dst_binding, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: RMS_NORM node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "RMS_NORM_F32" },
        { "name", tensor_name(node) },
        { "src", src_binding->handle },
        { "dst", dst_binding->handle },
        { "rows", node->ne[0] },
        { "cols", flattened_cols(node) },
        { "eps", op_param_f32(node, 0) },
        { "src_offset", src_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower RMS_NORM_F32 node=%s src=%s/%llu dst=%llu "
            "rows=%lld cols=%lld\n",
            tensor_name(node),
            tensor_name(src),
            static_cast<unsigned long long>(src_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(node->ne[0]),
            static_cast<long long>(flattened_cols(node)));
    }
    return true;
}

static bool append_matmul_q1a8_op(
    const ggml_tensor * node,
    nlohmann::json * ops,
    nlohmann::json * outputs) {
    if (!supports_matmul_q1a8(node)) {
        GGML_LOG_ERROR("pynq: MUL_MAT node %s is not a supported Q1A8 matmul\n", node->name);
        return false;
    }

    const ggml_tensor * weights = node->src[0];
    const ggml_tensor * acts = node->src[1];
    const RemoteBinding * weights_binding = find_tensor_binding(weights);
    const RemoteBinding * acts_binding = find_tensor_binding(acts);
    const RemoteBinding * dst_binding = find_tensor_binding(node);
    if (weights_binding == nullptr ||
        acts_binding == nullptr ||
        dst_binding == nullptr ||
        !remote_range_is_valid(*weights_binding, 0, ggml_nbytes(weights)) ||
        !remote_range_is_valid(*acts_binding, 0, ggml_nbytes(acts)) ||
        !remote_range_is_valid(*dst_binding, 0, ggml_nbytes(node))) {
        GGML_LOG_ERROR("pynq: MUL_MAT node %s is missing PYNQ tensor handles\n", node->name);
        return false;
    }

    ops->push_back({
        { "op", "MATMUL_Q1A8" },
        { "name", tensor_name(node) },
        { "weights", weights_binding->handle },
        { "acts", acts_binding->handle },
        { "dst", dst_binding->handle },
        { "rows", weights->ne[1] },
        { "cols", acts->ne[1] },
        { "k", weights->ne[0] },
        { "weights_offset", weights_binding->remote_offset },
        { "acts_offset", acts_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
    if (trace_enabled()) {
        tracef(
            "pynq trace: lower MATMUL_Q1A8 node=%s weights=%s/%llu "
            "acts=%s/%llu dst=%llu rows=%lld cols=%lld k=%lld\n",
            tensor_name(node),
            tensor_name(weights),
            static_cast<unsigned long long>(weights_binding->handle),
            tensor_name(acts),
            static_cast<unsigned long long>(acts_binding->handle),
            static_cast<unsigned long long>(dst_binding->handle),
            static_cast<long long>(weights->ne[1]),
            static_cast<long long>(acts->ne[1]),
            static_cast<long long>(weights->ne[0]));
    }
    return true;
}

static enum ggml_status backend_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * cgraph) {
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

        switch (node->op) {
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                if (!append_f32_binary_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CPY:
                if (!append_copy_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_RMS_NORM:
                if (!append_rms_norm_f32_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SCALE:
                if (!append_scale_f32_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_UNARY:
                if (!append_silu_f32_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GLU:
                if (!append_swiglu_f32_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (!append_matmul_q1a8_op(node, &ops, &outputs)) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("pynq: unsupported graph op %s in node %s\n",
                    ggml_op_name(node->op),
                    node->name);
                return GGML_STATUS_FAILED;
        }
    }

    if (ops.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    BackendContext * ctx = static_cast<BackendContext *>(backend->context);
    try {
        pynq::RpcClient rpc(ctx->endpoint);
        const size_t graph_call = g_trace_graph_calls.fetch_add(1) + 1;
        if (trace_enabled()) {
            tracef(
                "pynq trace: RUN_GRAPH #%zu submit ops=%zu outputs=%zu\n",
                graph_call,
                ops.size(),
                outputs.size());
        }
        const pynq::RpcResponse response = rpc.call(
            "RUN_GRAPH",
            {
                { "graph_version", k_graph_version },
                { "ops", ops },
                { "outputs", outputs },
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

static const ggml_backend_i backend_i = {
    /* .get_name                = */ backend_get_name,
    /* .free                    = */ backend_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ backend_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static bool hello_compatible(const pynq::RpcResponse & response) {
    return response.result.value("abi_version", -1) == k_runtime_abi_version &&
        response.result.value("server", "") == "bonsaid";
}

static ggml_backend_t init_backend(ggml_backend_dev_t device) {
    pynq::Endpoint endpoint;
    try {
        endpoint = pynq::endpoint_from_env();
        pynq::RpcClient rpc(endpoint);
        const pynq::RpcResponse hello = rpc.call("HELLO");
        if (!hello_compatible(hello)) {
            GGML_LOG_ERROR("pynq: incompatible bonsaid HELLO response\n");
            return nullptr;
        }
        if (trace_enabled()) {
            tracef(
                "pynq trace: HELLO endpoint=%s:%u memory=%s graph_ops=%s\n",
                endpoint.host.c_str(),
                static_cast<unsigned>(endpoint.port),
                hello.result.value("memory", nlohmann::json::object()).dump().c_str(),
                hello.result.value("graph_ops", nlohmann::json::array()).dump().c_str());
        }
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("pynq: bonsaid HELLO failed: %s\n", exc.what());
        return nullptr;
    }

    std::unique_ptr<BackendContext> ctx(new (std::nothrow) BackendContext { endpoint });
    if (ctx == nullptr) {
        return nullptr;
    }

    std::unique_ptr<ggml_backend> backend(new (std::nothrow) ggml_backend {
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_i,
        /* .device  = */ device,
        /* .context = */ ctx.get(),
    });
    if (backend == nullptr) {
        return nullptr;
    }

    ctx.release();
    return backend.release();
}

static const char * device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return k_backend_name;
}

static const char * device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return k_device_description;
}

static void device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
    try {
        const pynq::Endpoint endpoint = pynq::endpoint_from_env();
        tracef(
            "pynq trace: device memory query endpoint=%s:%u\n",
            endpoint.host.c_str(),
            static_cast<unsigned>(endpoint.port));
        pynq::RpcClient rpc(endpoint);
        const pynq::RpcResponse response = rpc.call("MEMORY");
        *free = memory_value(response.result, "free_bytes");
        *total = memory_value(response.result, "total_bytes");
        tracef(
            "pynq trace: device memory free=%.2f MiB total=%.2f MiB\n",
            mib(*free),
            mib(*total));
    } catch (const std::exception & exc) {
        tracef("pynq trace: device memory query failed: %s\n", exc.what());
    }
}

static enum ggml_backend_dev_type device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = device_get_name(dev);
    props->description = device_get_description(dev);
    props->type = device_get_type(dev);
    props->device_id = nullptr;
    device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return init_backend(dev);
}

static ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buffer_type = {
        /* .iface   = */ buffer_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    buffer_type.device = dev;
    return &buffer_type;
}

static bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (op != nullptr && is_metadata_op(op->op)) {
        return true;
    }
    return supports_raw_copy(op) ||
        supports_f32_binary(op) ||
        supports_scale_f32(op) ||
        supports_silu_f32(op) ||
        supports_swiglu_f32(op) ||
        supports_rms_norm_f32(op) ||
        supports_matmul_q1a8(op);
}

static bool device_supports_buft(
    ggml_backend_dev_t dev,
    ggml_backend_buffer_type_t buft) {
    return buft == device_get_buffer_type(dev);
}

static bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return supports_f32_binary(op) ||
        supports_scale_f32(op) ||
        supports_silu_f32(op) ||
        supports_swiglu_f32(op) ||
        supports_rms_norm_f32(op) ||
        supports_matmul_q1a8(op);
}

static const ggml_backend_device_i device_i = {
    /* .get_name             = */ device_get_name,
    /* .get_description      = */ device_get_description,
    /* .get_memory           = */ device_get_memory,
    /* .get_type             = */ device_get_type,
    /* .get_props            = */ device_get_props,
    /* .init_backend         = */ device_init_backend,
    /* .get_buffer_type      = */ device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ device_supports_op,
    /* .supports_buft        = */ device_supports_buft,
    /* .offload_op           = */ device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return k_backend_name;
}

static size_t reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ device_i,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    device.reg = reg;
    return &device;
}

static ggml_backend_feature g_features[] = {
    { "transport", "bonsaid-tcp" },
    { "buffer", "remote-tensor-handles" },
    { "graph_ops", "copy,matmul_q1a8,add_f32,mul_f32,scale_f32,silu_f32,swiglu_f32,rms_norm_f32" },
    { nullptr, nullptr },
};

static ggml_backend_feature * get_features(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_features;
}

static void * reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_get_features") == 0) {
        return reinterpret_cast<void *>(get_features);
    }
    return nullptr;
}

static const ggml_backend_reg_i reg_i = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

} // namespace

ggml_backend_reg_t ggml_backend_pynq_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_pynq_reg)
