#include "client.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace pynq {
namespace {

constexpr std::array<uint8_t, 4> k_magic = {'B', 'P', 'N', 'Q'};
constexpr uint16_t k_version = 1;
constexpr size_t k_header_size = 16;
constexpr size_t k_max_json_bytes = 1 * 1024 * 1024;
constexpr const char * k_default_host = "127.0.0.1";
constexpr uint16_t k_default_port = 50055;

std::string errno_text(const char * prefix) {
    std::ostringstream message;
    message << prefix << ": " << std::strerror(errno);
    return message.str();
}

void close_fd(int fd) {
    if (fd >= 0) {
        close(fd);
    }
}

void send_all(int fd, const void * data, size_t size) {
    const uint8_t * cursor = static_cast<const uint8_t *>(data);
    size_t remaining = size;
    while (remaining != 0) {
        const ssize_t sent = send(fd, cursor, remaining, 0);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            throw RpcError(errno_text("send failed"));
        }
        cursor += sent;
        remaining -= static_cast<size_t>(sent);
    }
}

void recv_all(int fd, void * data, size_t size) {
    uint8_t * cursor = static_cast<uint8_t *>(data);
    size_t remaining = size;
    while (remaining != 0) {
        const ssize_t received = recv(fd, cursor, remaining, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received < 0) {
            throw RpcError(errno_text("recv failed"));
        }
        if (received == 0) {
            throw RpcError("unexpected EOF while reading bonsaid frame");
        }
        cursor += received;
        remaining -= static_cast<size_t>(received);
    }
}

void write_u16(std::array<uint8_t, k_header_size> & header, size_t offset, uint16_t value) {
    const uint16_t encoded = htons(value);
    std::memcpy(header.data() + offset, &encoded, sizeof(encoded));
}

void write_u32(std::array<uint8_t, k_header_size> & header, size_t offset, uint32_t value) {
    const uint32_t encoded = htonl(value);
    std::memcpy(header.data() + offset, &encoded, sizeof(encoded));
}

uint16_t read_u16(const std::array<uint8_t, k_header_size> & header, size_t offset) {
    uint16_t encoded = 0;
    std::memcpy(&encoded, header.data() + offset, sizeof(encoded));
    return ntohs(encoded);
}

uint32_t read_u32(const std::array<uint8_t, k_header_size> & header, size_t offset) {
    uint32_t encoded = 0;
    std::memcpy(&encoded, header.data() + offset, sizeof(encoded));
    return ntohl(encoded);
}

int connect_endpoint(const Endpoint & endpoint) {
    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo * addresses = nullptr;
    const std::string port = std::to_string(endpoint.port);
    const int lookup = getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &addresses);
    if (lookup != 0) {
        throw RpcError("cannot resolve bonsaid endpoint " + endpoint.host + ":" + port +
            ": " + gai_strerror(lookup));
    }

    int fd = -1;
    for (addrinfo * address = addresses; address != nullptr; address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (connect(fd, address->ai_addr, address->ai_addrlen) == 0) {
            freeaddrinfo(addresses);
            return fd;
        }
        close_fd(fd);
        fd = -1;
    }

    freeaddrinfo(addresses);
    throw RpcError("cannot connect to bonsaid at " + endpoint.host + ":" + port);
}

uint16_t env_port() {
    const char * raw = std::getenv("PYNQ_BONSAID_PORT");
    if (raw == nullptr || raw[0] == '\0') {
        return k_default_port;
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' ||
        parsed == 0 || parsed > std::numeric_limits<uint16_t>::max()) {
        throw RpcError("invalid PYNQ_BONSAID_PORT");
    }
    return static_cast<uint16_t>(parsed);
}

std::string remote_error_text(const nlohmann::json & response) {
    const nlohmann::json error = response.value("error", nlohmann::json::object());
    const std::string code = error.value("code", "remote_error");
    const std::string message = error.value("message", "");
    return code + ": " + message;
}

} // namespace

Endpoint endpoint_from_env() {
    const char * host = std::getenv("PYNQ_BONSAID_HOST");
    return Endpoint {
        host != nullptr && host[0] != '\0' ? host : k_default_host,
        env_port(),
    };
}

RpcClient::RpcClient(const Endpoint & endpoint) : fd_(connect_endpoint(endpoint)) {
}

RpcClient::~RpcClient() {
    close_fd(fd_);
}

RpcResponse RpcClient::call(
    const std::string & op,
    const nlohmann::json & fields,
    const void * payload,
    size_t payload_size) {
    if (!fields.is_object()) {
        throw RpcError("RPC fields must be a JSON object");
    }
    if (payload_size != 0 && payload == nullptr) {
        throw RpcError("RPC payload pointer is null");
    }
    if (payload_size > std::numeric_limits<uint32_t>::max()) {
        throw RpcError("RPC payload exceeds frame limit");
    }

    const uint64_t request_id = next_id_++;
    nlohmann::json metadata = fields;
    metadata["id"] = request_id;
    metadata["op"] = op;
    const std::string metadata_text = metadata.dump();
    if (metadata_text.size() > k_max_json_bytes) {
        throw RpcError("RPC metadata exceeds frame limit");
    }

    std::array<uint8_t, k_header_size> header = {};
    std::copy(k_magic.begin(), k_magic.end(), header.begin());
    write_u16(header, 4, k_version);
    write_u16(header, 6, 0);
    write_u32(header, 8, static_cast<uint32_t>(metadata_text.size()));
    write_u32(header, 12, static_cast<uint32_t>(payload_size));
    send_all(fd_, header.data(), header.size());
    send_all(fd_, metadata_text.data(), metadata_text.size());
    if (payload_size != 0) {
        send_all(fd_, payload, payload_size);
    }

    recv_all(fd_, header.data(), header.size());
    if (!std::equal(k_magic.begin(), k_magic.end(), header.begin())) {
        throw RpcError("bonsaid response has bad frame magic");
    }
    if (read_u16(header, 4) != k_version) {
        throw RpcError("bonsaid response has unsupported protocol version");
    }
    if (read_u16(header, 6) != 0) {
        throw RpcError("bonsaid response has unsupported frame flags");
    }

    const size_t response_json_size = read_u32(header, 8);
    const size_t response_payload_size = read_u32(header, 12);
    if (response_json_size > k_max_json_bytes) {
        throw RpcError("bonsaid response metadata exceeds frame limit");
    }

    std::string response_text(response_json_size, '\0');
    if (!response_text.empty()) {
        recv_all(fd_, response_text.data(), response_text.size());
    }

    nlohmann::json response;
    try {
        response = nlohmann::json::parse(response_text);
    } catch (const nlohmann::json::exception & exc) {
        throw RpcError(std::string("invalid bonsaid response JSON: ") + exc.what());
    }
    if (!response.is_object()) {
        throw RpcError("bonsaid response metadata is not an object");
    }
    if (!response.contains("id") || response["id"] != request_id) {
        throw RpcError("bonsaid response id does not match request");
    }

    RpcResponse result;
    result.payload.resize(response_payload_size);
    if (!result.payload.empty()) {
        recv_all(fd_, result.payload.data(), result.payload.size());
    }

    if (!response.value("ok", false)) {
        throw RpcError(remote_error_text(response));
    }

    result.result = response.value("result", nlohmann::json::object());
    return result;
}

} // namespace pynq
