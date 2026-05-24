#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

// Structured NDJSON event log. Mirror of board/profiling/events.py.
//
// PYNQ_PROFILE controls emission:
//   unset / "0" / "false"   disabled
//   "1"                     stream events to stderr
//   any other value         treated as a file path; opened append, line-buffered
//
// Schema: {"t_us": <epoch micros>, "side": "host", "kind": "<name>", ...fields}

namespace pynq::events {

bool enabled();
void emit(const std::string & kind, const nlohmann::json & fields = nlohmann::json::object());

} // namespace pynq::events
