#pragma once

#include "driver.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Layer 2: Bonsai matmul lowering.
// This layer translates GGML_OP_MUL_MAT tensor metadata into Bonsai tile
// commands. It owns ggml shape/stride/quant details; drivers stay tensor-free.

namespace bonsai {

using i64 = int64_t;

struct MatMulJob {
    ggml_tensor * dst = nullptr;

    const ggml_tensor * src0 = nullptr;
    const ggml_tensor * src1 = nullptr;

    i64 k = 0;
    i64 n_rows = 0;
    i64 n_cols = 0;
    i64 n_b2 = 0;
    i64 n_b3 = 0;

    i64 src0_b2 = 0;
    i64 src0_b3 = 0;

    size_t src0_nb1 = 0;
    size_t src0_nb2 = 0;
    size_t src0_nb3 = 0;

    size_t src1_nb1 = 0;
    size_t src1_nb2 = 0;
    size_t src1_nb3 = 0;

    size_t dst_nb0 = 0;
    size_t dst_nb1 = 0;
    size_t dst_nb2 = 0;
    size_t dst_nb3 = 0;
};

bool make_matmul_job(ggml_tensor * dst, MatMulJob * job);
bool supports_matmul(const ggml_tensor * op);
bool run_bonsai_matmul(const MatMulJob & job, BonsaiDriver & driver, DriverKind driver_kind, int n_threads);

} // namespace bonsai
