#include "internal.h"
#include "trace.h"

#include "proto/ops.h"
#include "proto/q1a8_layout.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

namespace pynq {

namespace P = pynq::proto;

namespace {

// Per-ggml-buffer state. We give ggml a unique mmap'd "fake base" pointer
// so its bookkeeping has stable identity, then translate (base + offset)
// back to a remote tensor handle when ggml hands us a tensor. RPC goes
// through the shared process-wide client.
struct BufferContext {
    void * base = nullptr;
    std::size_t size = 0;
    uint64_t remote_handle = 0;
    std::size_t remote_nbytes = 0;
    std::unordered_map<const ggml_tensor *, RemoteBinding> bindings;
};

BufferContext * buffer_ctx(ggml_backend_buffer_t buffer) {
    return static_cast<BufferContext *>(buffer->context);
}

const RemoteBinding * find_binding(BufferContext * ctx, const ggml_tensor * tensor) {
    const auto it = ctx->bindings.find(tensor);
    return it == ctx->bindings.end() ? nullptr : &it->second;
}

// Q1_0 weight tensors are repacked at upload into the AXIS rowblock layout
// (board/kernels/pl/matmul_q1a8.py reads them that way). For tensors whose
// row count is a multiple of ROWS_PER_BLOCK the packed size equals
// ggml_nbytes, but partial trailing rowblocks need padding — e.g. Bonsai
// 1.7B's lm_head has 151669 rows, packed needs 3 extra zero-padded rows,
// adding 864 bytes. We override get_alloc_size to give every Q1_0 tensor a
// packed-size slot, and the binding records that size as well.
bool is_repackable_q1_0(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_Q1_0) return false;
    if (tensor->ne[0] <= 0 || tensor->ne[1] <= 0) return false;
    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) return false;
    if (tensor->ne[0] % BONSAI_Q1_BLOCK != 0) return false;
    if (!ggml_is_contiguous(tensor)) return false;
    return true;
}

std::size_t q1_0_packed_size(const ggml_tensor * tensor) {
    return bonsai_q1a8_packed_nbytes(
        static_cast<uint32_t>(tensor->ne[1]),
        static_cast<uint32_t>(tensor->ne[0]));
}

// Effective on-board allocation size for a tensor: packed size for Q1_0,
// ggml_nbytes for everything else.
std::size_t effective_alloc_size(const ggml_tensor * tensor) {
    if (is_repackable_q1_0(tensor)) {
        return q1_0_packed_size(tensor);
    }
    return ggml_nbytes(tensor);
}

bool should_repack_as_axis_q1a8(const ggml_tensor * tensor,
                                std::size_t offset, std::size_t size) {
    if (!is_repackable_q1_0(tensor)) return false;
    if (offset != 0) return false;
    return size == ggml_nbytes(tensor);
}

void upload_fill(uint64_t handle, std::size_t offset, std::size_t size, uint8_t value) {
    std::vector<uint8_t> fill(std::min(size, k_fill_chunk_size), value);
    std::size_t done = 0;
    while (done != size) {
        const std::size_t chunk = std::min(size - done, fill.size());
        shared_client().call(
            P::OP_UPLOAD_TENSOR,
            {
                { P::F_HANDLE, handle },
                { P::F_OFFSET, add_checked(offset, done) },
            },
            fill.data(),
            chunk);
        done += chunk;
    }
}

bool tensor_buffer_offset(
    const BufferContext * ctx,
    const ggml_tensor * tensor,
    std::size_t * offset) {
    if (ctx->size == 0) {
        *offset = 0;
        return ggml_nbytes(tensor) == 0;
    }
    if (ctx->base == nullptr || tensor->data == nullptr) {
        return false;
    }

    const auto base = reinterpret_cast<std::uintptr_t>(ctx->base);
    const auto data = reinterpret_cast<std::uintptr_t>(tensor->data);
    if (data < base) {
        return false;
    }

    const std::size_t local_offset = static_cast<std::size_t>(data - base);
    const std::size_t nbytes = effective_alloc_size(tensor);
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

void free_buffer_allocation(BufferContext * ctx) {
    if (ctx->remote_handle != 0) {
        try {
            shared_client().call(P::OP_FREE_TENSOR, { { P::F_HANDLE, ctx->remote_handle } });
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

void * reserve_fake_base(std::size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void * base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return base == MAP_FAILED ? nullptr : base;
}

void release_fake_base(void * base, std::size_t size) {
    if (base != nullptr && size != 0) {
        munmap(base, size);
    }
}

void pynq_buffer_free(ggml_backend_buffer_t buffer) {
    BufferContext * ctx = buffer_ctx(buffer);
    free_buffer_allocation(ctx);
    release_fake_base(ctx->base, ctx->size);
    delete ctx;
}

void * pynq_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer_ctx(buffer)->base;
}

bool is_pynq_buffer(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->iface.get_base == pynq_buffer_get_base;
}

enum ggml_status pynq_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
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

        // Use packed size for Q1_0 tensors so the binding's bounds match
        // what we'll actually upload, and so subsequent tensors land at the
        // arena offset ggml computed via get_alloc_size.
        const std::size_t nbytes = effective_alloc_size(tensor);
        std::size_t remote_offset = 0;
        if (!tensor_buffer_offset(ctx, tensor, &remote_offset)) {
            GGML_LOG_ERROR("pynq: tensor %s is outside its PYNQ buffer arena\n", tensor->name);
            return GGML_STATUS_FAILED;
        }
        if (nbytes != 0 && ctx->remote_handle == 0) {
            GGML_LOG_ERROR("pynq: tensor %s has no backing PYNQ buffer allocation\n", tensor->name);
            return GGML_STATUS_FAILED;
        }

        RemoteBinding binding { ctx->remote_handle, ctx->remote_nbytes, nbytes, remote_offset };
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

void pynq_buffer_memset_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    uint8_t value,
    std::size_t offset,
    std::size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid memset range for tensor %s\n", tensor->name);
        return;
    }
    try {
        upload_fill(binding->handle, add_checked(binding->remote_offset, offset), size, value);
        if (trace_enabled()) {
            const std::size_t total =
                trace_counters().uploaded_bytes.fetch_add(size) + size;
            tracef(
                "pynq trace: memset name=%s handle=%llu offset=%zu bytes=%zu "
                "value=%u uploaded_total=%.2f MiB\n",
                tensor_name(tensor),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                size,
                static_cast<unsigned>(value),
                mib(total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("memset", tensor, exc);
    }
}

void pynq_buffer_set_tensor(
    ggml_backend_buffer_t buffer,
    ggml_tensor * tensor,
    const void * data,
    std::size_t offset,
    std::size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid upload range for tensor %s\n", tensor->name);
        return;
    }

    // Q1_0 weight tensor uploaded whole: repack into AXIS rowblock layout.
    // The packed bytes may be longer than the Q1_0 source if rows isn't a
    // multiple of ROWS_PER_BLOCK (partial trailing rowblock is zero-padded).
    // get_alloc_size and the binding already reserve the packed size, so
    // uploading want > size is safe.
    std::vector<uint8_t> repacked;
    const void * upload_data = data;
    std::size_t  upload_size = size;
    const bool repack = should_repack_as_axis_q1a8(tensor, offset, size);
    if (repack) {
        const uint32_t k    = static_cast<uint32_t>(tensor->ne[0]);
        const uint32_t rows = static_cast<uint32_t>(tensor->ne[1]);
        const size_t   want = bonsai_q1a8_packed_nbytes(rows, k);
        try {
            repacked.resize(want);
        } catch (const std::bad_alloc &) {
            GGML_LOG_ERROR("pynq: cannot allocate %zu-byte repack scratch for %s\n",
                want, tensor_name(tensor));
            return;
        }
        const int rc = bonsai_q1a8_pack_weights(
            static_cast<const uint8_t *>(data), rows, k, repacked.data());
        if (rc != 0) {
            GGML_LOG_ERROR("pynq: Q1_0 repack failed for tensor %s rc=%d\n",
                tensor_name(tensor), rc);
            return;
        }
        upload_data = repacked.data();
        upload_size = want;
        if (trace_enabled()) {
            tracef("pynq trace: repack name=%s rows=%u k=%u q1_0=%zu packed=%zu\n",
                tensor_name(tensor), rows, k, size, want);
        }
    } else if (tensor->type == GGML_TYPE_Q1_0 && trace_enabled()) {
        // Surface partial Q1_0 uploads so we notice if llama.cpp ever starts
        // doing chunked weight loads — that would leave a mixed-layout tensor.
        tracef("pynq trace: WARN Q1_0 partial upload name=%s offset=%zu size=%zu "
               "(not repacked; on-board layout may be inconsistent)\n",
            tensor_name(tensor), offset, size);
    }

    try {
        shared_client().call(
            P::OP_UPLOAD_TENSOR,
            {
                { P::F_HANDLE, binding->handle },
                { P::F_OFFSET, add_checked(binding->remote_offset, offset) },
            },
            upload_data,
            upload_size);
        if (trace_enabled()) {
            const std::size_t total =
                trace_counters().uploaded_bytes.fetch_add(upload_size) + upload_size;
            tracef(
                "pynq trace: upload name=%s type=%s handle=%llu offset=%zu "
                "bytes=%zu repack=%s uploaded_total=%.2f MiB\n",
                tensor_name(tensor),
                ggml_type_name(tensor->type),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                upload_size,
                repack ? "axis_q1a8" : "raw",
                mib(total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("upload", tensor, exc);
    }
}

void pynq_buffer_get_tensor(
    ggml_backend_buffer_t buffer,
    const ggml_tensor * tensor,
    void * data,
    std::size_t offset,
    std::size_t size) {
    BufferContext * ctx = buffer_ctx(buffer);
    const RemoteBinding * binding = find_binding(ctx, tensor);
    if (binding == nullptr || !remote_range_is_valid(*binding, offset, size)) {
        GGML_LOG_ERROR("pynq: invalid download range for tensor %s\n", tensor->name);
        return;
    }
    try {
        const pynq::RpcResponse response = shared_client().call(
            P::OP_DOWNLOAD_TENSOR,
            {
                { P::F_HANDLE, binding->handle },
                { P::F_OFFSET, add_checked(binding->remote_offset, offset) },
                { P::F_SIZE, size },
            });
        if (response.payload.size() != size) {
            throw pynq::RpcError("DOWNLOAD_TENSOR returned an unexpected payload size");
        }
        std::memcpy(data, response.payload.data(), size);
        if (trace_enabled()) {
            const std::size_t total =
                trace_counters().downloaded_bytes.fetch_add(size) + size;
            tracef(
                "pynq trace: download name=%s type=%s handle=%llu offset=%zu "
                "bytes=%zu downloaded_total=%.2f MiB\n",
                tensor_name(tensor),
                ggml_type_name(tensor->type),
                static_cast<unsigned long long>(binding->handle),
                add_checked(binding->remote_offset, offset),
                size,
                mib(total));
        }
    } catch (const std::exception & exc) {
        log_rpc_failure("download", tensor, exc);
    }
}

bool pynq_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

void pynq_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    BufferContext * ctx = buffer_ctx(buffer);
    if (ctx->remote_handle == 0 || ctx->remote_nbytes == 0) {
        return;
    }
    try {
        upload_fill(ctx->remote_handle, 0, ctx->remote_nbytes, value);
        if (trace_enabled()) {
            const std::size_t total =
                trace_counters().uploaded_bytes.fetch_add(ctx->remote_nbytes) + ctx->remote_nbytes;
            tracef(
                "pynq trace: clear handle=%llu bytes=%zu value=%u uploaded_total=%.2f MiB\n",
                static_cast<unsigned long long>(ctx->remote_handle),
                ctx->remote_nbytes,
                static_cast<unsigned>(value),
                mib(total));
        }
    } catch (const std::exception & exc) {
        GGML_LOG_ERROR("pynq: clear failed for handle %llu: %s\n",
            static_cast<unsigned long long>(ctx->remote_handle),
            exc.what());
    }
}

void pynq_buffer_reset(ggml_backend_buffer_t buffer) {
    buffer_ctx(buffer)->bindings.clear();
}

const ggml_backend_buffer_i buffer_i = {
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

// -- buffer_type ----------------------------------------------------------

const char * buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "PYNQ_REMOTE";
}

ggml_backend_buffer_t buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, std::size_t size) {
    std::unique_ptr<BufferContext> ctx;
    try {
        ctx.reset(new BufferContext());
        ctx->base = reserve_fake_base(size);
        ctx->size = size;
        if (size != 0 && ctx->base == nullptr) {
            GGML_LOG_ERROR("pynq: cannot reserve %zu bytes of local tensor address space\n", size);
            return nullptr;
        }

        if (size != 0) {
            const pynq::RpcResponse response = shared_client().call(
                P::OP_ALLOC_TENSOR,
                {
                    { P::F_NBYTES, size },
                    { P::F_SHAPE, nlohmann::json::array({ size }) },
                    { P::F_DTYPE, "u8" },
                    { P::F_USAGE, "ggml-buffer" },
                    { P::F_LAYOUT, "ggml-buffer" },
                    { P::F_ALIGNMENT, k_alignment },
                });
            const nlohmann::json & remote = response.result.at(P::F_TENSOR);
            ctx->remote_handle = remote.at(P::F_HANDLE).get<uint64_t>();
            ctx->remote_nbytes = remote.at(P::F_NBYTES).get<std::size_t>();
        }

        if (trace_enabled()) {
            const std::size_t total =
                trace_counters().allocated_bytes.fetch_add(ctx->remote_nbytes) + ctx->remote_nbytes;
            tracef(
                "pynq trace: buffer reserve bytes=%zu fake_base=%p handle=%llu "
                "remote_bytes=%zu allocated_total=%.2f MiB\n",
                size,
                ctx->base,
                static_cast<unsigned long long>(ctx->remote_handle),
                ctx->remote_nbytes,
                mib(total));
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

std::size_t buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return k_alignment;
}

std::size_t memory_value(const nlohmann::json & result, const char * key) {
    return result.at(P::F_MEMORY).at(key).get<std::size_t>();
}

std::size_t buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    try {
        const pynq::RpcResponse response = shared_client().call(P::OP_MEMORY);
        const std::size_t free_bytes = memory_value(response.result, P::F_FREE_BYTES);
        tracef("pynq trace: buffer max_size free=%.2f MiB\n", mib(free_bytes));
        return free_bytes;
    } catch (const std::exception & exc) {
        tracef("pynq trace: buffer max_size query failed: %s\n", exc.what());
        return std::numeric_limits<std::size_t>::max();
    }
}

std::size_t buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // Q1_0 weight tensors need extra room when rows isn't a multiple of
    // ROWS_PER_BLOCK (the packed layout pads the trailing partial rowblock).
    return effective_alloc_size(tensor);
}

bool buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

const ggml_backend_buffer_type_i buffer_type_i = {
    /* .get_name       = */ buffer_type_get_name,
    /* .alloc_buffer   = */ buffer_type_alloc_buffer,
    /* .get_alignment  = */ buffer_type_get_alignment,
    /* .get_max_size   = */ buffer_type_get_max_size,
    /* .get_alloc_size = */ buffer_type_get_alloc_size,
    /* .is_host        = */ buffer_type_is_host,
};

} // namespace

const RemoteBinding * find_tensor_binding(const ggml_tensor * tensor) {
    if (tensor == nullptr || !is_pynq_buffer(tensor->buffer)) {
        return nullptr;
    }
    return find_binding(buffer_ctx(tensor->buffer), tensor);
}

ggml_backend_buffer_type_t pynq_buffer_type() {
    static ggml_backend_buffer_type type = {
        /* .iface   = */ buffer_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &type;
}

} // namespace pynq
