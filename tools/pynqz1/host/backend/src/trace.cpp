#include "trace.h"

#include "internal.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// PYNQ_TRACE accepts either:
//   "0", "false", unset → disabled
//   "1"                 → human-readable lines to stderr
//   anything else       → treated as a file path; opened append, line-buffered
// Same convention used by PYNQ_PROFILE on the board (see board/profiling).

namespace pynq {

namespace {

std::FILE * resolve_sink() {
    const char * value = std::getenv("PYNQ_TRACE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
        std::strcmp(value, "false") == 0 || std::strcmp(value, "no") == 0) {
        return nullptr;
    }
    if (std::strcmp(value, "1") == 0) {
        return stderr;
    }
    std::FILE * fp = std::fopen(value, "a");
    if (fp == nullptr) {
        std::fprintf(stderr, "pynq: PYNQ_TRACE='%s' could not be opened, falling back to stderr\n", value);
        return stderr;
    }
    std::setvbuf(fp, nullptr, _IOLBF, 0);
    return fp;
}

std::FILE * sink() {
    static std::FILE * fp = resolve_sink();
    return fp;
}

} // namespace

bool trace_enabled() {
    return sink() != nullptr;
}

void tracef(const char * format, ...) {
    std::FILE * fp = sink();
    if (fp == nullptr) {
        return;
    }
    std::va_list args;
    va_start(args, format);
    std::vfprintf(fp, format, args);
    va_end(args);
    std::fflush(fp);
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
