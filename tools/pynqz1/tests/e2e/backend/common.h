#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

// Shared fixtures and golden-data helpers for backend e2e tests. Each
// test_*.cpp file builds on these and exposes one or more ``run_*``
// functions that ``main.cpp`` invokes in sequence.

namespace pynq_e2e {

constexpr int64_t k_matmul_k = 128;
constexpr int64_t k_matmul_rows = 3;
constexpr int64_t k_matmul_cols = 2;
constexpr int64_t k_glue_rows = 4;
constexpr int64_t k_glue_cols = 2;

// Tolerance comparison — exact for memcpy paths, relative for F32 math.
bool same_floats(
    const std::vector<float> & lhs,
    const std::vector<float> & rhs,
    float tolerance = 1e-6f);

float fp16_roundtrip(float value);
float silu(float value);

// Deterministic test inputs.
std::vector<float> make_matmul_weights();
std::vector<float> make_matmul_acts();
std::vector<float> make_glue_input();
std::vector<float> make_glue_bias();
std::vector<float> make_swiglu_up();

// Golden outputs that match the C kernel exactly (Q1/Q8 quantization
// boundaries + fp16 scale roundtrip — must agree bit-for-bit with the
// daemon's matmul, hence the explicit reimplementation here).
std::vector<float> expected_matmul(
    const std::vector<float> & weights,
    const std::vector<float> & acts);
std::vector<float> expected_glue_output(
    const std::vector<float> & input,
    const std::vector<float> & bias);
std::vector<float> expected_swiglu_output(
    const std::vector<float> & gate,
    const std::vector<float> & up);

bool quantize_matmul_weights(
    const std::vector<float> & weights,
    std::vector<uint8_t> * q1_weights);

} // namespace pynq_e2e

// Per-test entrypoints. Each returns true on success, false on failure
// (after printing a diagnostic to stderr). ``main.cpp`` aggregates them.
bool run_upload_download(ggml_backend_t backend, ggml_backend_dev_t dev);
bool run_copy(ggml_backend_t backend, ggml_backend_dev_t dev);
bool run_matmul_direct(ggml_backend_t backend, ggml_backend_dev_t dev);
bool run_swiglu_direct(ggml_backend_t backend, ggml_backend_dev_t dev);
bool run_matmul_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend);
bool run_glue_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend);
bool run_swiglu_scheduler(ggml_backend_t backend, ggml_backend_t cpu_backend);
bool run_multi_extent(ggml_backend_t backend, ggml_backend_dev_t dev);
