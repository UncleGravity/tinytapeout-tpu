#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "client.h"
#include "ggml-backend.h"

// Cross-TU types and helpers shared by device.cpp, buffer.cpp, and lowering.cpp.
//
// BufferContext stays private to buffer.cpp — lowering only needs to look
// up a RemoteBinding for a tensor, exposed via find_tensor_binding().

namespace pynq {

constexpr const char * k_backend_name = "PYNQ";
constexpr const char * k_device_description = "PYNQ-Z1 bonsaid remote tensor backend";
constexpr std::size_t k_alignment = 64;
constexpr std::size_t k_fill_chunk_size = 1 * 1024 * 1024;

struct BackendContext {
    pynq::Endpoint endpoint;
};

struct RemoteBinding {
    uint64_t handle = 0;
    std::size_t handle_nbytes = 0;
    std::size_t tensor_nbytes = 0;
    std::size_t remote_offset = 0;
};

inline std::size_t add_checked(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw pynq::RpcError("tensor range overflows size_t");
    }
    return lhs + rhs;
}

inline bool remote_range_is_valid(const RemoteBinding & binding, std::size_t offset, std::size_t size) {
    try {
        const std::size_t tensor_end = add_checked(offset, size);
        const std::size_t remote_end = add_checked(binding.remote_offset, tensor_end);
        return tensor_end <= binding.tensor_nbytes && remote_end <= binding.handle_nbytes;
    } catch (const pynq::RpcError &) {
        return false;
    }
}

inline const char * tensor_name(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->name[0] != '\0' ? tensor->name : "(unnamed)";
}

// Implemented in buffer.cpp — returns nullptr if the tensor is not from a
// PYNQ buffer or has not been bound yet.
const RemoteBinding * find_tensor_binding(const ggml_tensor * tensor);

// Implemented in device.cpp — the backend GUID.
ggml_guid_t backend_guid();

// Implemented in buffer.cpp — the buffer-type singleton, also referenced
// by the device interface for ``device_get_buffer_type``.
ggml_backend_buffer_type_t pynq_buffer_type();

} // namespace pynq
