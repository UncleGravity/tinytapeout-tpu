#pragma once

#include <atomic>
#include <cstddef>
#include <exception>

#include "ggml-impl.h"

// PYNQ_TRACE=1 host-side instrumentation. Cheap when disabled (single
// cached env check); printf-style line per event when enabled. Counters
// are atomic so they survive whatever threading ggml schedules around us.

namespace pynq {

bool trace_enabled();
void tracef(const char * format, ...);
double mib(std::size_t nbytes);

struct TraceCounters {
    std::atomic<std::size_t> allocated_bytes { 0 };
    std::atomic<std::size_t> uploaded_bytes { 0 };
    std::atomic<std::size_t> downloaded_bytes { 0 };
    std::atomic<std::size_t> graph_calls { 0 };
};

TraceCounters & trace_counters();

void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc);

} // namespace pynq
