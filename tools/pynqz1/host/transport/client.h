#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pynq {

struct Endpoint {
    std::string host;
    uint16_t port = 0;
};

struct RpcResponse {
    nlohmann::json result;
    std::vector<uint8_t> payload;
};

class RpcError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Endpoint endpoint_from_env();

// Long-lived RPC client. Holds one socket to the daemon and reuses it
// across calls. ``call`` is mutex-protected so multiple ggml code paths
// (model load + graph compute) can share a single instance. On send/recv
// failure the socket is dropped; the next call transparently reconnects.
class RpcClient {
public:
    explicit RpcClient(Endpoint endpoint);
    ~RpcClient();

    RpcClient(const RpcClient &) = delete;
    RpcClient & operator=(const RpcClient &) = delete;

    RpcResponse call(
        const std::string & op,
        const nlohmann::json & fields = nlohmann::json::object(),
        const void * payload = nullptr,
        size_t payload_size = 0);

    const Endpoint & endpoint() const { return endpoint_; }

private:
    void ensure_connected();
    void close_socket() noexcept;
    RpcResponse call_locked(
        const std::string & op,
        const nlohmann::json & fields,
        const void * payload,
        size_t payload_size);

    Endpoint endpoint_;
    int fd_ = -1;
    uint64_t next_id_ = 1;
    std::mutex mu_;
};

// Process-wide client, lazy-constructed against ``endpoint_from_env()`` on
// first call. Use everywhere instead of per-call ``RpcClient(endpoint)``.
RpcClient & shared_client();

} // namespace pynq
