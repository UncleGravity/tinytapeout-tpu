#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace pynq {

struct Endpoint {
    std::string host;
    uint16_t port;
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

class RpcClient {
public:
    explicit RpcClient(const Endpoint & endpoint);
    ~RpcClient();

    RpcClient(const RpcClient &) = delete;
    RpcClient & operator=(const RpcClient &) = delete;

    RpcResponse call(
        const std::string & op,
        const nlohmann::json & fields = nlohmann::json::object(),
        const void * payload = nullptr,
        size_t payload_size = 0);

private:
    int fd_ = -1;
    uint64_t next_id_ = 1;
};

} // namespace pynq
