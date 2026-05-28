#include "trace.h"

#include "internal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// -- unsupported-op census ------------------------------------------------

namespace {

std::mutex & census_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, std::size_t> & census_counts() {
    static std::unordered_map<std::string, std::size_t> m;
    return m;
}

std::string op_label(const ggml_tensor * op) {
    if (op == nullptr) return "(null)";
    const char * base = ggml_op_name(op->op);
    if (op->op == GGML_OP_UNARY) {
        return std::string(base) + ":" + ggml_unary_op_name(ggml_get_unary_op(op));
    }
    if (op->op == GGML_OP_GLU) {
        return std::string(base) + ":" + ggml_glu_op_name(ggml_get_glu_op(op));
    }
    if (op->op == GGML_OP_MUL_MAT) {
        // Split MUL_MAT by operand types so we see Q1_0×F32 vs F32×F32 separately.
        const ggml_tensor * w = op->src[0];
        const ggml_tensor * a = op->src[1];
        if (w != nullptr && a != nullptr) {
            return std::string(base) + ":"
                + ggml_type_name(w->type) + "x" + ggml_type_name(a->type);
        }
    }
    return base;
}

} // namespace

void record_unsupported_op(const ggml_tensor * op) {
    const std::string label = op_label(op);
    std::lock_guard<std::mutex> lock(census_mutex());
    census_counts()[label]++;
}

void dump_unsupported_op_census() {
    std::lock_guard<std::mutex> lock(census_mutex());
    auto & counts = census_counts();
    if (counts.empty()) return;
    std::vector<std::pair<std::string, std::size_t>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    std::fprintf(stderr, "\n=== pynq unsupported-op census (split offenders) ===\n");
    for (const auto & [name, count] : sorted) {
        std::fprintf(stderr, "  %-40s %zu\n", name.c_str(), count);
    }
    std::fprintf(stderr, "===\n\n");
    counts.clear();
}

void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc) {
    GGML_LOG_ERROR("pynq: %s failed for tensor %s: %s\n",
        action,
        tensor_name(tensor),
        exc.what());
}

} // namespace pynq
