#include "ggml-bonsai.h"

#include "matmul.h"
#include "transport.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

// ggml backend plumbing.
// This file only registers the device, accepts graph splits, and hands
// supported MUL_MAT nodes to the lowering layer.

namespace {

constexpr const char * k_backend_name       = "Bonsai";
constexpr const char * k_device_description = "Bonsai accelerator scaffold";
constexpr const char * k_set_threads_proc   = "ggml_backend_set_n_threads";
constexpr const char * k_get_features_proc  = "ggml_backend_get_features";

struct BackendContext {
    // n_threads is accepted from ggml's set_n_threads hook for API
    // compatibility but currently ignored — the matmul lowering runs
    // cells serially against a single Transport instance.
    int n_threads = GGML_DEFAULT_N_THREADS;
    uint64_t n_graphs = 0;
    uint64_t n_mul_mat = 0;
    int trace_limit = 0;
    // -2 = no MUL_MAT seen yet, -1 = last weight wasn't a per-block tensor
    // (e.g. token_embd / output / lm_head), 0..L-1 = transformer block index.
    int last_layer_idx = -2;
    std::chrono::steady_clock::time_point t_start =
        std::chrono::steady_clock::now();
    std::unique_ptr<bonsai::Transport> transport;
};

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static int env_positive_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

static int get_trace_limit() {
    return env_enabled("BONSAI_TRACE_MATMUL")
        ? env_positive_int("BONSAI_TRACE_LIMIT", 16)
        : 0;
}

// llama.cpp names per-block weights as "blk.<N>.foo.weight". Anything that
// doesn't match (token_embd, output norm, lm_head, etc.) returns -1.
static int parse_layer_idx(const char * name) {
    if (name == nullptr) return -1;
    constexpr char prefix[] = "blk.";
    constexpr size_t prefix_len = sizeof(prefix) - 1;
    if (std::strncmp(name, prefix, prefix_len) != 0) return -1;
    const char * p = name + prefix_len;
    if (*p < '0' || *p > '9') return -1;
    int idx = 0;
    while (*p >= '0' && *p <= '9') {
        idx = idx * 10 + (*p - '0');
        ++p;
    }
    return *p == '.' ? idx : -1;
}

// Emit one stderr line whenever the per-layer index changes, so a user
// staring at a multi-hour forward pass can tell where they are. Uses
// fprintf+fflush directly (rather than GGML_LOG_INFO) so it's not subject
// to llama.cpp's --verbosity gating or any host-side log buffering.
static void log_layer_progress(BackendContext & ctx, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int layer_idx = parse_layer_idx(src0 ? src0->name : nullptr);
    if (layer_idx == ctx.last_layer_idx) {
        return;
    }

    const double elapsed_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ctx.t_start).count();
    const unsigned long long n_done = (unsigned long long) ctx.n_mul_mat;

    if (layer_idx >= 0) {
        std::fprintf(stderr,
            "bonsai: starting layer %d (after %llu MUL_MAT, %.1fs elapsed)\n",
            layer_idx, n_done, elapsed_s);
    } else {
        std::fprintf(stderr,
            "bonsai: starting %s (after %llu MUL_MAT, %.1fs elapsed)\n",
            (src0 && src0->name[0]) ? src0->name : "(unnamed)",
            n_done, elapsed_s);
    }
    std::fflush(stderr);
    ctx.last_layer_idx = layer_idx;
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

static void trace_matmul(const BackendContext & ctx, const ggml_tensor * dst) {
    if (ctx.trace_limit <= 0 || (int) ctx.n_mul_mat > ctx.trace_limit) {
        return;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_LOG_INFO(
            "bonsai: MUL_MAT #%llu %s: src0=%s [%lld,%lld,%lld,%lld] %s, "
            "src1=%s [%lld,%lld,%lld,%lld] %s -> [%lld,%lld,%lld,%lld] %s via %s\n",
            (unsigned long long) ctx.n_mul_mat,
            dst->name,
            src0->name,
            (long long) src0->ne[0], (long long) src0->ne[1],
            (long long) src0->ne[2], (long long) src0->ne[3],
            ggml_type_name(src0->type),
            src1->name,
            (long long) src1->ne[0], (long long) src1->ne[1],
            (long long) src1->ne[2], (long long) src1->ne[3],
            ggml_type_name(src1->type),
            (long long) dst->ne[0], (long long) dst->ne[1],
            (long long) dst->ne[2], (long long) dst->ne[3],
            ggml_type_name(dst->type),
            ctx.transport->name());
}

static enum ggml_status compute_graph(ggml_backend_t backend, ggml_cgraph * cgraph) {
    BackendContext * ctx = (BackendContext *) backend->context;
    ++ctx->n_graphs;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (is_metadata_op(node->op)) {
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT) {
            GGML_LOG_ERROR("bonsai: unsupported op %s in graph split\n", ggml_op_desc(node));
            return GGML_STATUS_FAILED;
        }

        bonsai::MatMulJob job;
        if (!bonsai::make_matmul_job(node, &job)) {
            GGML_LOG_ERROR("bonsai: unsupported MUL_MAT shape or type for node %s\n", node->name);
            return GGML_STATUS_FAILED;
        }

        log_layer_progress(*ctx, node);
        ++ctx->n_mul_mat;
        trace_matmul(*ctx, node);
        if (!bonsai::run_bonsai_matmul(job, *ctx->transport)) {
            GGML_LOG_ERROR("bonsai: transport %s failed while computing node %s\n",
                    ctx->transport->name(), node->name);
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const char * backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return k_backend_name;
}

static void backend_free(ggml_backend_t backend) {
    BackendContext * ctx = (BackendContext *) backend->context;
    if (ctx != nullptr && ctx->trace_limit > 0) {
        GGML_LOG_INFO("bonsai: computed %llu MUL_MAT node(s) across %llu graph split(s) via %s\n",
                (unsigned long long) ctx->n_mul_mat,
                (unsigned long long) ctx->n_graphs,
                ctx->transport ? ctx->transport->name() : "(none)");
    }

    delete ctx;
    delete backend;
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
    /* .graph_compute           = */ compute_graph,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t backend_guid() {
    static ggml_guid guid = {
        0x62, 0x6f, 0x6e, 0x73, 0x61, 0x69, 0x2d, 0x74,
        0x74, 0x70, 0x75, 0x2d, 0x30, 0x30, 0x30, 0x31,
    };
    return &guid;
}

static ggml_backend_t init_backend() {
    std::unique_ptr<BackendContext> ctx(new (std::nothrow) BackendContext);
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->trace_limit = get_trace_limit();
    ctx->transport   = bonsai::create_usb_transport();
    if (ctx->transport == nullptr) {
        GGML_LOG_ERROR("bonsai: USB transport unavailable; backend will not load "
                       "(ggml scheduler will route MUL_MAT to other backends)\n");
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ backend_guid(),
        /* .iface   = */ backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_bonsai_reg(), 0),
        /* .context = */ ctx.get(),
    };

    if (backend == nullptr) {
        return nullptr;
    }

    ctx.release();
    return backend;
}

static void set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(backend != nullptr);
    GGML_ASSERT(ggml_guid_matches(backend->guid, backend_guid()));

    BackendContext * ctx = (BackendContext *) backend->context;
    ctx->n_threads = n_threads;
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
}

static enum ggml_backend_dev_type device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = device_get_name(dev);
    props->description = device_get_description(dev);
    props->type        = device_get_type(dev);
    props->device_id   = nullptr;
    device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return init_backend();
}

static ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    if (is_metadata_op(op->op)) {
        return true;
    }
    if (!bonsai::supports_matmul(op)) {
        return false;
    }
    // The W1A8 fabric only handles 1-bit-weight matmuls. FP src0 tensors
    // (e.g. attention Q@K^T, attn@V) would be destroyed by binarization, so we
    // leave them for the CPU backend via the scheduler.
    return op->src[0]->type == GGML_TYPE_Q1_0;
}

static bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && device_supports_op(dev, op);
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
    /* .buffer_from_host_ptr = */ device_buffer_from_host_ptr,
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
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &device;
}

static ggml_backend_feature g_features[] = {
    { "mul_mat",   "bonsai-plan-lowering" },
    { "transport", "usb" },
    { "buffer",    "host" },
    { nullptr,     nullptr },
};

static ggml_backend_feature * get_features(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_features;
}

static void * reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, k_set_threads_proc) == 0) {
        return (void *) set_n_threads;
    }

    if (std::strcmp(name, k_get_features_proc) == 0) {
        return (void *) get_features;
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

ggml_backend_reg_t ggml_backend_bonsai_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ reg_i,
        /* .context     = */ nullptr,
    };

    return &reg;
}

// Export ggml_backend_init(), the standard dynamic-loader entry point that
// returns this backend's registry when llama.cpp loads libggml-bonsai.so.
GGML_BACKEND_DL_IMPL(ggml_backend_bonsai_reg)
