#include "ggml-bonsai.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <thread>
#include <vector>

// This backend is currently a scheduler/driver scaffold:
// - ggml tensors stay in host memory via the CPU buffer type
// - the device advertises itself as an accelerator for MUL_MAT
// - graph_compute dispatches supported matmuls to a replaceable driver
//
// The CPU driver below is intentionally plain. It is the host-side stand-in for
// a future RTL/FPGA command driver.
namespace {

constexpr const char * k_backend_name        = "Bonsai";
constexpr const char * k_device_description  = "Bonsai fake accelerator";
constexpr const char * k_set_threads_proc    = "ggml_backend_set_n_threads";
constexpr const char * k_get_features_proc   = "ggml_backend_get_features";

using i64 = int64_t;

struct MatMulJob {
    ggml_tensor * dst = nullptr;

    const ggml_tensor * src0 = nullptr;
    const ggml_tensor * src1 = nullptr;

    i64 k = 0;
    i64 n_rows = 0;
    i64 n_cols = 0;
    i64 n_b2 = 0;
    i64 n_b3 = 0;

    i64 src0_b2 = 0;
    i64 src0_b3 = 0;

    size_t src0_nb1 = 0;
    size_t src0_nb2 = 0;
    size_t src0_nb3 = 0;

    size_t src1_nb1 = 0;
    size_t src1_nb2 = 0;
    size_t src1_nb3 = 0;

    size_t dst_nb0 = 0;
    size_t dst_nb1 = 0;
    size_t dst_nb2 = 0;
    size_t dst_nb3 = 0;
};

class CpuReferenceDriver {
public:
    void compute_mul_mat(const MatMulJob & job, int n_threads) const {
        const i64 work = job.n_rows * job.n_cols * job.n_b2 * job.n_b3;
        if (work == 0) {
            return;
        }

        n_threads = std::max(1, n_threads);
        n_threads = std::min<int>(n_threads, (int) work);

        auto worker = [&](int ith) {
            std::vector<float> src0_scratch;
            std::vector<float> src1_scratch;

            for (i64 iw = ith; iw < work; iw += n_threads) {
                compute_cell(job, iw, src0_scratch, src1_scratch);
            }
        };

        if (n_threads == 1) {
            worker(0);
            return;
        }

        std::vector<std::thread> threads;
        threads.reserve(n_threads - 1);
        for (int ith = 1; ith < n_threads; ++ith) {
            threads.emplace_back(worker, ith);
        }

        worker(0);

        for (std::thread & thread : threads) {
            thread.join();
        }
    }

private:
    static const float * row_to_float(
            const ggml_tensor * tensor,
            const void * data,
            i64 n,
            std::vector<float> & scratch) {
        if (tensor->type == GGML_TYPE_F32) {
            return (const float *) data;
        }

        scratch.resize(n);
        ggml_get_type_traits(tensor->type)->to_float(data, scratch.data(), n);
        return scratch.data();
    }

    static float dot(const float * x, const float * y, i64 n) {
        float sum = 0.0f;
        for (i64 i = 0; i < n; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    static void compute_cell(
            const MatMulJob & job,
            i64 linear_index,
            std::vector<float> & src0_scratch,
            std::vector<float> & src1_scratch) {
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

        const float * x = row_to_float(job.src0, src0_row, job.k, src0_scratch);
        const float * y = row_to_float(job.src1, src1_row, job.k, src1_scratch);
        *dst_cell = dot(x, y, job.k);
    }
};

struct BackendContext {
    int n_threads = GGML_DEFAULT_N_THREADS;
    uint64_t n_graphs = 0;
    uint64_t n_mul_mat = 0;
    int trace_limit = 0;
    CpuReferenceDriver driver;
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

static bool type_can_convert_to_float(ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->to_float != nullptr;
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

static bool make_matmul_job(ggml_tensor * dst, MatMulJob * job) {
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

static bool supports_matmul(const ggml_tensor * op) {
    return make_matmul_job((ggml_tensor *) op, nullptr);
}

static void trace_matmul(const BackendContext & ctx, const ggml_tensor * dst) {
    if (ctx.trace_limit <= 0 || (int) ctx.n_mul_mat > ctx.trace_limit) {
        return;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_LOG_INFO(
            "bonsai: MUL_MAT #%llu %s: src0=%s [%lld,%lld,%lld,%lld] %s, "
            "src1=%s [%lld,%lld,%lld,%lld] %s -> [%lld,%lld,%lld,%lld] %s\n",
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
            ggml_type_name(dst->type));
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

        MatMulJob job;
        if (!make_matmul_job(node, &job)) {
            GGML_LOG_ERROR("bonsai: unsupported MUL_MAT shape or type for node %s\n", node->name);
            return GGML_STATUS_FAILED;
        }

        ++ctx->n_mul_mat;
        trace_matmul(*ctx, node);
        ctx->driver.compute_mul_mat(job, ctx->n_threads);
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
        GGML_LOG_INFO("bonsai: computed %llu MUL_MAT node(s) across %llu graph split(s)\n",
                (unsigned long long) ctx->n_mul_mat,
                (unsigned long long) ctx->n_graphs);
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
    return is_metadata_op(op->op) || supports_matmul(op);
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
    { "mul_mat", "host-f32-reference" },
    { "buffer",  "host" },
    { nullptr,   nullptr },
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

/*
Macro from `ggml-backend-impl.h`.
When `GGML_BACKEND_DL` is enabled, it expands into an exported C ABI function
named ggml_backend_init(), that calls ggml_backend_bonsai_reg() and returns
the `ggml_backend_reg_t`.
*/
GGML_BACKEND_DL_IMPL(ggml_backend_bonsai_reg)
