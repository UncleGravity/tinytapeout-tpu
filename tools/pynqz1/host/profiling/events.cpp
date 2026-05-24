#include "events.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace pynq::events {

namespace {

std::FILE * resolve_sink() {
    const char * value = std::getenv("PYNQ_PROFILE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 || std::strcmp(value, "no") == 0 ||
        std::strcmp(value, "off") == 0) {
        return nullptr;
    }
    if (std::strcmp(value, "1") == 0) {
        return stderr;
    }
    std::FILE * fp = std::fopen(value, "a");
    if (fp == nullptr) {
        std::fprintf(stderr, "pynq: PYNQ_PROFILE='%s' could not be opened, falling back to stderr\n", value);
        return stderr;
    }
    std::setvbuf(fp, nullptr, _IOLBF, 0);
    return fp;
}

std::FILE * sink() {
    static std::FILE * fp = resolve_sink();
    return fp;
}

std::mutex & write_lock() {
    static std::mutex m;
    return m;
}

std::int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

bool enabled() {
    return sink() != nullptr;
}

void emit(const std::string & kind, const nlohmann::json & fields) {
    std::FILE * fp = sink();
    if (fp == nullptr) {
        return;
    }
    nlohmann::json record = fields;
    record["t_us"] = now_us();
    record["side"] = "host";
    record["kind"] = kind;
    const std::string line = record.dump();

    std::lock_guard<std::mutex> lock(write_lock());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fputc('\n', fp);
    std::fflush(fp);
}

} // namespace pynq::events
