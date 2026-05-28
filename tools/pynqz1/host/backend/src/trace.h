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

// Census of ops the scheduler asked about that we returned false for. Used
// to size the cost-of-adding-support for graph fusion (Phase 2). Always
// records, even with PYNQ_TRACE off — dump happens at backend_free via
// dump_unsupported_op_census() so the histogram is visible exactly once.
void record_unsupported_op(const ggml_tensor * op);
void dump_unsupported_op_census();

void log_rpc_failure(const char * action, const ggml_tensor * tensor, const std::exception & exc);

} // namespace pynq
