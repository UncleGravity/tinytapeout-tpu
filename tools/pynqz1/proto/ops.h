#pragma once

// PYNQ-Z1 runtime wire schema — keep in sync with proto/ops.py.
// Field constants are bare strings so they compose with nlohmann::json.

#include <cstdint>

namespace pynq::proto {

inline constexpr int ABI_VERSION   = 1;
inline constexpr int GRAPH_VERSION = 1;
inline constexpr std::uint16_t DEFAULT_PORT = 50055;
inline constexpr const char SERVER_NAME[]   = "bonsaid";

// Top-level RPCs
inline constexpr const char OP_HELLO[]            = "HELLO";
inline constexpr const char OP_MEMORY[]           = "MEMORY";
inline constexpr const char OP_ALLOC_TENSOR[]     = "ALLOC_TENSOR";
inline constexpr const char OP_UPLOAD_TENSOR[]    = "UPLOAD_TENSOR";
inline constexpr const char OP_DOWNLOAD_TENSOR[]  = "DOWNLOAD_TENSOR";
inline constexpr const char OP_FREE_TENSOR[]      = "FREE_TENSOR";
inline constexpr const char OP_RUN_GRAPH[]        = "RUN_GRAPH";

// Graph ops
inline constexpr const char GOP_COPY[]         = "COPY";
inline constexpr const char GOP_MATMUL_Q1A8[]  = "MATMUL_Q1A8";
inline constexpr const char GOP_ADD_F32[]      = "ADD_F32";
inline constexpr const char GOP_MUL_F32[]      = "MUL_F32";
inline constexpr const char GOP_SCALE_F32[]    = "SCALE_F32";
inline constexpr const char GOP_SILU_F32[]     = "SILU_F32";
inline constexpr const char GOP_SWIGLU_F32[]   = "SWIGLU_F32";
inline constexpr const char GOP_RMS_NORM_F32[] = "RMS_NORM_F32";
inline constexpr const char GOP_ROPE_F32[]     = "ROPE_F32";
inline constexpr const char GOP_FLASH_ATTN_EXT_F32[] = "FLASH_ATTN_EXT_F32";
inline constexpr const char GOP_GET_ROWS[]     = "GET_ROWS";
inline constexpr const char GOP_SET_ROWS[]     = "SET_ROWS";

// Envelope
inline constexpr const char F_ID[]      = "id";
inline constexpr const char F_OP[]      = "op";
inline constexpr const char F_OK[]      = "ok";
inline constexpr const char F_RESULT[]  = "result";
inline constexpr const char F_ERROR[]   = "error";
inline constexpr const char F_CODE[]    = "code";
inline constexpr const char F_MESSAGE[] = "message";

// Tensor allocation
inline constexpr const char F_NBYTES[]       = "nbytes";
inline constexpr const char F_SHAPE[]        = "shape";
inline constexpr const char F_DTYPE[]        = "dtype";
inline constexpr const char F_USAGE[]        = "usage";
inline constexpr const char F_LAYOUT[]       = "layout";
inline constexpr const char F_ALIGNMENT[]    = "alignment";
inline constexpr const char F_TENSOR[]       = "tensor";
inline constexpr const char F_HANDLE[]       = "handle";
inline constexpr const char F_OFFSET[]       = "offset";
inline constexpr const char F_SIZE[]         = "size";
inline constexpr const char F_EXTENT_COUNT[] = "extent_count";

// Memory accounting
inline constexpr const char F_MEMORY[]       = "memory";
inline constexpr const char F_TOTAL_BYTES[]  = "total_bytes";
inline constexpr const char F_FREE_BYTES[]   = "free_bytes";
inline constexpr const char F_USED_BYTES[]   = "used_bytes";
inline constexpr const char F_SLAB_COUNT[]   = "slab_count";
inline constexpr const char F_TENSOR_COUNT[] = "tensor_count";

// HELLO
inline constexpr const char F_ABI_VERSION[]  = "abi_version";
inline constexpr const char F_SERVER[]       = "server";
inline constexpr const char F_OVERLAY_ID[]   = "overlay_id";
inline constexpr const char F_CAPABILITIES[] = "capabilities";
inline constexpr const char F_GRAPH_OPS[]    = "graph_ops";

// RUN_GRAPH
inline constexpr const char F_GRAPH_VERSION[] = "graph_version";
inline constexpr const char F_OPS[]           = "ops";
inline constexpr const char F_OUTPUTS[]       = "outputs";
inline constexpr const char F_OP_COUNT[]      = "op_count";
inline constexpr const char F_COUNTERS[]      = "counters";
inline constexpr const char F_PS_OPS[]        = "ps_ops";
inline constexpr const char F_PL_OPS[]        = "pl_ops";
inline constexpr const char F_BYTES_READ[]    = "bytes_read";
inline constexpr const char F_BYTES_WRITTEN[] = "bytes_written";
inline constexpr const char F_ELAPSED_US[]    = "elapsed_us";

// Graph op fields
inline constexpr const char F_SRC[]             = "src";
inline constexpr const char F_DST[]             = "dst";
inline constexpr const char F_SRC_OFFSET[]      = "src_offset";
inline constexpr const char F_DST_OFFSET[]      = "dst_offset";
inline constexpr const char F_SRC0[]            = "src0";
inline constexpr const char F_SRC1[]            = "src1";
inline constexpr const char F_SRC0_OFFSET[]     = "src0_offset";
inline constexpr const char F_SRC1_OFFSET[]     = "src1_offset";
inline constexpr const char F_SRC1_BROADCAST[]  = "src1_broadcast";
inline constexpr const char F_WEIGHTS[]         = "weights";
inline constexpr const char F_ACTS[]            = "acts";
inline constexpr const char F_WEIGHTS_OFFSET[]  = "weights_offset";
inline constexpr const char F_ACTS_OFFSET[]     = "acts_offset";
inline constexpr const char F_ROWS[]            = "rows";
inline constexpr const char F_COLS[]            = "cols";
inline constexpr const char F_K[]               = "k";
inline constexpr const char F_ELEMENTS[]        = "elements";
inline constexpr const char F_SCALE[]           = "scale";
inline constexpr const char F_BIAS[]            = "bias";
inline constexpr const char F_EPS[]             = "eps";
inline constexpr const char F_NAME[]            = "name";

// ROPE / FLASH_ATTN_EXT fields
inline constexpr const char F_POSITIONS[]       = "positions";
inline constexpr const char F_POSITIONS_OFFSET[] = "positions_offset";
inline constexpr const char F_HEAD_DIM[]        = "head_dim";
inline constexpr const char F_N_HEAD[]          = "n_head";
inline constexpr const char F_N_HEAD_KV[]       = "n_head_kv";
inline constexpr const char F_N_TOKEN[]         = "n_token";
inline constexpr const char F_N_KV[]            = "n_kv";
inline constexpr const char F_N_DIMS[]          = "n_dims";
inline constexpr const char F_MODE[]            = "mode";
inline constexpr const char F_FREQ_BASE[]       = "freq_base";
inline constexpr const char F_FREQ_SCALE[]      = "freq_scale";
inline constexpr const char F_ATTN_FACTOR[]     = "attn_factor";
inline constexpr const char F_EXT_FACTOR[]      = "ext_factor";
inline constexpr const char F_BETA_FAST[]       = "beta_fast";
inline constexpr const char F_BETA_SLOW[]       = "beta_slow";
inline constexpr const char F_N_CTX_ORIG[]      = "n_ctx_orig";

// FLASH_ATTN_EXT additional fields
inline constexpr const char F_K_TENSOR[]        = "k_tensor";
inline constexpr const char F_K_OFFSET[]        = "k_offset";
inline constexpr const char F_V_TENSOR[]        = "v_tensor";
inline constexpr const char F_V_OFFSET[]        = "v_offset";
inline constexpr const char F_MASK[]            = "mask";
inline constexpr const char F_MASK_OFFSET[]     = "mask_offset";
inline constexpr const char F_HAS_MASK[]        = "has_mask";
inline constexpr const char F_HEAD_DIM_Q[]      = "head_dim_q";
inline constexpr const char F_HEAD_DIM_V[]      = "head_dim_v";
inline constexpr const char F_MAX_BIAS[]        = "max_bias";
inline constexpr const char F_LOGIT_SOFTCAP[]   = "logit_softcap";

// K/V/mask are typically views into a persistent KV cache, so they aren't
// fully contiguous (each dim is regular-strided, but heads are spaced by
// ctx_max not n_kv). The kernel needs the actual strides; nb0 is implicit
// (= type_size: 4 for Q, 2 for K/V/mask in F16). Strides are in BYTES.
inline constexpr const char F_Q_NB1[]           = "q_nb1";
inline constexpr const char F_Q_NB2[]           = "q_nb2";
inline constexpr const char F_K_NB1[]           = "k_nb1";
inline constexpr const char F_K_NB2[]           = "k_nb2";
inline constexpr const char F_V_NB1[]           = "v_nb1";
inline constexpr const char F_V_NB2[]           = "v_nb2";
inline constexpr const char F_MASK_NB1[]        = "mask_nb1";
inline constexpr const char F_DST_NB1[]         = "dst_nb1";
inline constexpr const char F_DST_NB2[]         = "dst_nb2";

// GET_ROWS / SET_ROWS fields. Both ops have:
//   src0: the row-bearing tensor (embedding table for GET_ROWS,
//         dst for SET_ROWS' "from" — see below)
//   src1: integer indices (i32 for GET_ROWS, i32/i64 for SET_ROWS)
//   dst : the F32 output for GET_ROWS, the F16 destination for SET_ROWS
//
// SET_ROWS in ggml is structured as: dst.src[0]=values(F32), dst.src[1]=indices,
// dst=destination(F16). The op writes src0 rows into dst at indices given by
// src1, converting F32→F16. We mirror this with F_SRC0 + F_INDICES + F_DST.
inline constexpr const char F_INDICES[]         = "indices";
inline constexpr const char F_INDICES_OFFSET[]  = "indices_offset";
inline constexpr const char F_INDICES_TYPE[]    = "indices_type";   // "i32" | "i64"
inline constexpr const char F_SRC0_TYPE[]       = "src0_type";      // "f32" | "f16" | "q1_0"
inline constexpr const char F_DST_TYPE[]        = "dst_type";       // "f32" | "f16"
inline constexpr const char F_N_INDICES[]       = "n_indices";
inline constexpr const char F_SRC0_NB1[]        = "src0_nb1";
inline constexpr const char F_SRC0_NB2[]        = "src0_nb2";
inline constexpr const char F_SRC0_NB3[]        = "src0_nb3";
inline constexpr const char F_INDICES_NB1[]     = "indices_nb1";
inline constexpr const char F_INDICES_NB2[]     = "indices_nb2";
inline constexpr const char F_DST_NB3[]         = "dst_nb3";
inline constexpr const char F_NE10[]            = "ne10";  // indices ne0
inline constexpr const char F_NE11[]            = "ne11";  // indices ne1
inline constexpr const char F_NE12[]            = "ne12";  // indices ne2
inline constexpr const char F_NE01[]            = "ne01";  // src0 ne1 (row count)
inline constexpr const char F_NE02[]            = "ne02";  // src0 ne2
inline constexpr const char F_NE03[]            = "ne03";  // src0 ne3

// Error codes
inline constexpr const char ERR_INVALID_REQUEST[]            = "invalid_request";
inline constexpr const char ERR_OUT_OF_MEMORY[]              = "out_of_memory";
inline constexpr const char ERR_OUT_OF_BOUNDS[]              = "out_of_bounds";
inline constexpr const char ERR_UNKNOWN_TENSOR[]             = "unknown_tensor";
inline constexpr const char ERR_UNSUPPORTED_OP[]             = "unsupported_op";
inline constexpr const char ERR_UNSUPPORTED_GRAPH_VERSION[]  = "unsupported_graph_version";
inline constexpr const char ERR_PROTOCOL[]                   = "protocol_error";
inline constexpr const char ERR_INTERNAL[]                   = "internal_error";
inline constexpr const char ERR_REMOTE[]                     = "remote_error";

// Block sizes
inline constexpr int Q1_BLOCK       = 128;
inline constexpr int Q1_BLOCK_BYTES = 18;
inline constexpr int Q8_BLOCK       = 32;

} // namespace pynq::proto
