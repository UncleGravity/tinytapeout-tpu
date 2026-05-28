#include "ggml-pynq.h"

#include "internal.h"
#include "trace.h"

#include "proto/ops.h"

#include "ggml-backend-impl.h"

#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace pynq {

namespace P = pynq::proto;

// Implemented in lowering.cpp.
bool device_supports_op_impl(const ggml_tensor * op);
bool device_offload_op_impl(const ggml_tensor * op);
enum ggml_status backend_graph_compute_impl(ggml_backend_t backend, ggml_cgraph * cgraph);

namespace {

const char * backend_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return k_backend_name;
}

void backend_free(ggml_backend_t backend) {
    dump_unsupported_op_census();
    delete static_cast<BackendContext *>(backend->context);
    delete backend;
}

const ggml_backend_i backend_i = {
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
    /* .graph_compute           = */ backend_graph_compute_impl,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool hello_compatible(const pynq::RpcResponse & response) {
    return response.result.value(P::F_ABI_VERSION, -1) == P::ABI_VERSION &&
        response.result.value(P::F_SERVER, "") == P::SERVER_NAME;
}

ggml_backend_t init_backend(ggml_backend_dev_t device) {
    try {
        const pynq::RpcResponse hello = shared_client().call(P::OP_HELLO);
        if (!hello_compatible(hello)) {
            GGML_LOG_ERROR("pynq: incompatible bonsaid HELLO response\n");
            return nullptr;
        }
        if (trace_enabled()) {
            const Endpoint & ep = shared_client().endpoint();
            tracef(
                "pynq trace: HELLO endpoint=%s:%u memory=%s graph_ops=%s\n",
                ep.host.c_str(),
                static_cast<unsigned>(ep.port),
                hello.result.value(P::F_MEMORY, nlohmann::json::object()).dump().c_str(),
                hello.result.value(P::F_GRAPH_OPS, nlohmann::json::array()).dump().c_str());
        }
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("pynq: bonsaid HELLO failed: %s\n", exc.what());
        return nullptr;
    }

    std::unique_ptr<BackendContext> ctx(new (std::nothrow) BackendContext { shared_client().endpoint() });
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

// -- device interface -----------------------------------------------------

std::size_t memory_value(const nlohmann::json & result, const char * key) {
    return result.at(P::F_MEMORY).at(key).get<std::size_t>();
}

const char * device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return k_backend_name;
}

const char * device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return k_device_description;
}

void device_get_memory(ggml_backend_dev_t dev, std::size_t * free, std::size_t * total) {
    GGML_UNUSED(dev);
    *free = 0;
    *total = 0;
    try {
        const pynq::RpcResponse response = shared_client().call(P::OP_MEMORY);
        *free = memory_value(response.result, P::F_FREE_BYTES);
        *total = memory_value(response.result, P::F_TOTAL_BYTES);
        tracef("pynq trace: device memory free=%.2f MiB total=%.2f MiB\n",
            mib(*free), mib(*total));
    } catch (const std::exception & exc) {
        tracef("pynq trace: device memory query failed: %s\n", exc.what());
    }
}

enum ggml_backend_dev_type device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
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

ggml_backend_t device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return init_backend(dev);
}

ggml_backend_buffer_type_t device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_buffer_type_t buft = pynq_buffer_type();
    buft->device = dev;
    return buft;
}

bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return device_supports_op_impl(op);
}

bool device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == device_get_buffer_type(dev);
}

bool device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    return device_offload_op_impl(op);
}

const ggml_backend_device_i device_i = {
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

// -- registry -------------------------------------------------------------

const char * reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return k_backend_name;
}

std::size_t reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t reg_get_device(ggml_backend_reg_t reg, std::size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device device = {
        /* .iface   = */ device_i,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    device.reg = reg;
    return &device;
}

ggml_backend_feature g_features[] = {
    { "transport", "bonsaid-tcp" },
    { "buffer", "remote-tensor-handles" },
    { "graph_ops", "copy,matmul_q1a8,add_f32,mul_f32,scale_f32,silu_f32,swiglu_f32,rms_norm_f32,rope_f32,flash_attn_ext_f32,get_rows,set_rows" },
    { nullptr, nullptr },
};

ggml_backend_feature * get_features(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_features;
}

void * reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_get_features") == 0) {
        return reinterpret_cast<void *>(get_features);
    }
    return nullptr;
}

const ggml_backend_reg_i reg_i = {
    /* .get_name         = */ reg_get_name,
    /* .get_device_count = */ reg_get_device_count,
    /* .get_device       = */ reg_get_device,
    /* .get_proc_address = */ reg_get_proc_address,
};

} // namespace

ggml_guid_t backend_guid() {
    static ggml_guid guid = {
        0x70, 0x79, 0x6e, 0x71, 0x2d, 0x62, 0x6f, 0x6e,
        0x73, 0x61, 0x69, 0x2d, 0x7a, 0x31, 0x30, 0x31,
    };
    return &guid;
}

} // namespace pynq

ggml_backend_reg_t ggml_backend_pynq_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ pynq::reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_pynq_reg)
