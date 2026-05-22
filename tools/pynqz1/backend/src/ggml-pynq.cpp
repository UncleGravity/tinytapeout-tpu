#include "ggml-pynq.h"

#include "rpc.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

struct BackendContext {
    pynq::Endpoint endpoint;
};

struct RemoteBinding {
    uint64_t handle = 0;
    size_t handle_nbytes = 0;
    size_t tensor_nbytes = 0;
    size_t remote_offset = 0;
};

struct RemoteAllocation {
    uint64_t handle = 0;
    size_t nbytes = 0;
};

struct BufferContext {
    explicit BufferContext(pynq::Endpoint endpoint) : rpc(endpoint) {
    }

    pynq::RpcClient rpc;
    void * base = nullptr;
    size_t size = 0;
    std::unordered_map<const ggml_tensor *, RemoteBinding> bindings;
    std::vector<RemoteAllocation> allocations;
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

static nlohmann::json tensor_shape(const ggml_tensor * tensor) {
    nlohmann::json shape = nlohmann::json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        shape.push_back(tensor->ne[i]);
    }
    return shape;
}

static void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc) {
    GGML_LOG_ERROR("pynq: %s failed for tensor %s: %s\n",
        action,
        tensor != nullptr && tensor->name[0] != '\0' ? tensor->name : "(unnamed)",
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

static void free_allocations(BufferContext * ctx) {
    for (const RemoteAllocation & allocation : ctx->allocations) {
        try {
            ctx->rpc.call("FREE_TENSOR", { { "handle", allocation.handle } });
        } catch (const std::exception & exc) {
            GGML_LOG_WARN("pynq: FREE_TENSOR failed for handle %llu: %s\n",
                static_cast<unsigned long long>(allocation.handle),
                exc.what());
        }
    }
    ctx->allocations.clear();
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
    free_allocations(ctx);
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
            return GGML_STATUS_SUCCESS;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        const pynq::RpcResponse response = ctx->rpc.call(
            "ALLOC_TENSOR",
            {
                { "nbytes", nbytes },
                { "shape", tensor_shape(tensor) },
                { "dtype", ggml_type_name(tensor->type) },
                { "usage", "ggml" },
                { "layout", "ggml" },
                { "alignment", k_alignment },
            });
        const nlohmann::json & remote = response.result.at("tensor");
        const uint64_t handle = remote.at("handle").get<uint64_t>();
        const size_t remote_nbytes = remote.at("nbytes").get<size_t>();
        ctx->bindings.emplace(tensor, RemoteBinding {
            handle,
            remote_nbytes,
            nbytes,
            0,
        });
        ctx->allocations.push_back(RemoteAllocation { handle, remote_nbytes });
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
    for (const RemoteAllocation & allocation : ctx->allocations) {
        try {
            upload_fill(ctx, allocation.handle, 0, allocation.nbytes, value);
        } catch (const std::exception & exc) {
            GGML_LOG_ERROR("pynq: clear failed for handle %llu: %s\n",
                static_cast<unsigned long long>(allocation.handle),
                exc.what());
        }
    }
}

static void pynq_buffer_reset(ggml_backend_buffer_t buffer) {
    free_allocations(buffer_ctx(buffer));
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
    try {
        std::unique_ptr<BufferContext> ctx(new BufferContext(pynq::endpoint_from_env()));
        ctx->base = reserve_fake_base(size);
        ctx->size = size;
        if (size != 0 && ctx->base == nullptr) {
            GGML_LOG_ERROR("pynq: cannot reserve %zu bytes of local tensor address space\n", size);
            return nullptr;
        }

        return ggml_backend_buffer_init(buft, buffer_i, ctx.release(), size);
    } catch (const std::exception & exc) {
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
        pynq::RpcClient rpc(pynq::endpoint_from_env());
        const pynq::RpcResponse response = rpc.call("MEMORY");
        return memory_value(response.result, "free_bytes");
    } catch (const std::exception &) {
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
        { "src", src_binding->handle },
        { "dst", dst_binding->handle },
        { "nbytes", nbytes },
        { "src_offset", src_binding->remote_offset },
        { "dst_offset", dst_binding->remote_offset },
    });
    outputs->push_back(dst_binding->handle);
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

        if (node->op != GGML_OP_CPY) {
            GGML_LOG_ERROR("pynq: unsupported graph op %s in node %s\n",
                ggml_op_name(node->op),
                node->name);
            return GGML_STATUS_FAILED;
        }

        if (!append_copy_op(node, &ops, &outputs)) {
            return GGML_STATUS_FAILED;
        }
    }

    if (ops.empty()) {
        return GGML_STATUS_SUCCESS;
    }

    BackendContext * ctx = static_cast<BackendContext *>(backend->context);
    try {
        pynq::RpcClient rpc(ctx->endpoint);
        rpc.call(
            "RUN_GRAPH",
            {
                { "graph_version", k_graph_version },
                { "ops", ops },
                { "outputs", outputs },
            });
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
        pynq::RpcClient rpc(pynq::endpoint_from_env());
        const pynq::RpcResponse response = rpc.call("MEMORY");
        *free = memory_value(response.result, "free_bytes");
        *total = memory_value(response.result, "total_bytes");
    } catch (const std::exception &) {
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
    return supports_raw_copy(op);
}

static bool device_supports_buft(
    ggml_backend_dev_t dev,
    ggml_backend_buffer_type_t buft) {
    return buft == device_get_buffer_type(dev);
}

static bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
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
    { "graph_ops", "copy" },
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
