#include "trace.h"

#include "internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace pynq {

bool trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("PYNQ_TRACE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void tracef(const char * format, ...) {
    if (!trace_enabled()) {
        return;
    }
    std::va_list args;
    va_start(args, format);
    std::vfprintf(stderr, format, args);
    va_end(args);
    std::fflush(stderr);
}

double mib(std::size_t nbytes) {
    return static_cast<double>(nbytes) / (1024.0 * 1024.0);
}

TraceCounters & trace_counters() {
    static TraceCounters counters;
    return counters;
}

void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc) {
    GGML_LOG_ERROR("pynq: %s failed for tensor %s: %s\n",
        action,
        tensor_name(tensor),
        exc.what());
}

} // namespace pynq
