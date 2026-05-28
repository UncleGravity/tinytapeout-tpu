"""PYNQ-Z1 runtime wire schema.

Single source of truth for the JSON field names, op names, and version
constants exchanged between the host backend (``libggml-pynq.so``) and
the board daemon (``bonsaid``).

Keep in sync with ``proto/ops.h`` — verified by ``proto/tests/test_parity.py``.
"""

from __future__ import annotations

ABI_VERSION = 1
GRAPH_VERSION = 1
DEFAULT_PORT = 50055
SERVER_NAME = "bonsaid"

# Top-level RPCs
OP_HELLO = "HELLO"
OP_MEMORY = "MEMORY"
OP_ALLOC_TENSOR = "ALLOC_TENSOR"
OP_UPLOAD_TENSOR = "UPLOAD_TENSOR"
OP_DOWNLOAD_TENSOR = "DOWNLOAD_TENSOR"
OP_FREE_TENSOR = "FREE_TENSOR"
OP_RUN_GRAPH = "RUN_GRAPH"

RPC_OPS = (
    OP_HELLO,
    OP_MEMORY,
    OP_ALLOC_TENSOR,
    OP_UPLOAD_TENSOR,
    OP_DOWNLOAD_TENSOR,
    OP_FREE_TENSOR,
    OP_RUN_GRAPH,
)

# Graph ops carried in RUN_GRAPH.ops[]
GOP_COPY = "COPY"
GOP_MATMUL_Q1A8 = "MATMUL_Q1A8"
GOP_ADD_F32 = "ADD_F32"
GOP_MUL_F32 = "MUL_F32"
GOP_SCALE_F32 = "SCALE_F32"
GOP_SILU_F32 = "SILU_F32"
GOP_SWIGLU_F32 = "SWIGLU_F32"
GOP_RMS_NORM_F32 = "RMS_NORM_F32"
GOP_ROPE_F32 = "ROPE_F32"
GOP_FLASH_ATTN_EXT_F32 = "FLASH_ATTN_EXT_F32"

GRAPH_OPS = (
    GOP_COPY,
    GOP_MATMUL_Q1A8,
    GOP_ADD_F32,
    GOP_MUL_F32,
    GOP_SCALE_F32,
    GOP_SILU_F32,
    GOP_SWIGLU_F32,
    GOP_RMS_NORM_F32,
    GOP_ROPE_F32,
    GOP_FLASH_ATTN_EXT_F32,
)

# Envelope
F_ID = "id"
F_OP = "op"
F_OK = "ok"
F_RESULT = "result"
F_ERROR = "error"
F_CODE = "code"
F_MESSAGE = "message"

# Tensor allocation
F_NBYTES = "nbytes"
F_SHAPE = "shape"
F_DTYPE = "dtype"
F_USAGE = "usage"
F_LAYOUT = "layout"
F_ALIGNMENT = "alignment"
F_TENSOR = "tensor"
F_HANDLE = "handle"
F_OFFSET = "offset"
F_SIZE = "size"
F_EXTENT_COUNT = "extent_count"

# Memory accounting
F_MEMORY = "memory"
F_TOTAL_BYTES = "total_bytes"
F_FREE_BYTES = "free_bytes"
F_USED_BYTES = "used_bytes"
F_SLAB_COUNT = "slab_count"
F_TENSOR_COUNT = "tensor_count"

# HELLO
F_ABI_VERSION = "abi_version"
F_SERVER = "server"
F_OVERLAY_ID = "overlay_id"
F_CAPABILITIES = "capabilities"
F_GRAPH_OPS = "graph_ops"

# RUN_GRAPH
F_GRAPH_VERSION = "graph_version"
F_OPS = "ops"
F_OUTPUTS = "outputs"
F_OP_COUNT = "op_count"
F_COUNTERS = "counters"
F_PS_OPS = "ps_ops"
F_PL_OPS = "pl_ops"
F_BYTES_READ = "bytes_read"
F_BYTES_WRITTEN = "bytes_written"
F_ELAPSED_US = "elapsed_us"

# Graph op fields
F_SRC = "src"
F_DST = "dst"
F_SRC_OFFSET = "src_offset"
F_DST_OFFSET = "dst_offset"
F_SRC0 = "src0"
F_SRC1 = "src1"
F_SRC0_OFFSET = "src0_offset"
F_SRC1_OFFSET = "src1_offset"
F_SRC1_BROADCAST = "src1_broadcast"
F_WEIGHTS = "weights"
F_ACTS = "acts"
F_WEIGHTS_OFFSET = "weights_offset"
F_ACTS_OFFSET = "acts_offset"
F_ROWS = "rows"
F_COLS = "cols"
F_K = "k"
F_ELEMENTS = "elements"
F_SCALE = "scale"
F_BIAS = "bias"
F_EPS = "eps"
F_NAME = "name"

# ROPE / FLASH_ATTN_EXT
F_POSITIONS = "positions"
F_POSITIONS_OFFSET = "positions_offset"
F_HEAD_DIM = "head_dim"
F_N_HEAD = "n_head"
F_N_HEAD_KV = "n_head_kv"
F_N_TOKEN = "n_token"
F_N_KV = "n_kv"
F_N_DIMS = "n_dims"
F_MODE = "mode"
F_FREQ_BASE = "freq_base"
F_FREQ_SCALE = "freq_scale"

F_K_TENSOR = "k_tensor"
F_K_OFFSET = "k_offset"
F_V_TENSOR = "v_tensor"
F_V_OFFSET = "v_offset"
F_MASK = "mask"
F_MASK_OFFSET = "mask_offset"
F_HAS_MASK = "has_mask"
F_HEAD_DIM_V = "head_dim_v"
F_MAX_BIAS = "max_bias"
F_LOGIT_SOFTCAP = "logit_softcap"

# Error codes returned by the daemon. Hosts should switch on these, not on
# the human-readable message.
ERR_INVALID_REQUEST = "invalid_request"
ERR_OUT_OF_MEMORY = "out_of_memory"
ERR_OUT_OF_BOUNDS = "out_of_bounds"
ERR_UNKNOWN_TENSOR = "unknown_tensor"
ERR_UNSUPPORTED_OP = "unsupported_op"
ERR_UNSUPPORTED_GRAPH_VERSION = "unsupported_graph_version"
ERR_PROTOCOL = "protocol_error"
ERR_INTERNAL = "internal_error"
ERR_REMOTE = "remote_error"

# Numeric block sizes — needed by both host (lowering checks) and board
# (matmul kernel layout).
Q1_BLOCK = 128
Q1_BLOCK_BYTES = 18
Q8_BLOCK = 32
