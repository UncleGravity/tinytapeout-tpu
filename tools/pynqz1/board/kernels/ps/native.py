"""PS-resident kernels.

Each ``Kernel`` here owns a single ctypes function pointer plus the
glue that converts allocator-resident bytes into the C ABI and back.
No profile dict is threaded through — kernels report their own
``read`` / ``compute`` / ``write`` spans via the ``Timer`` they receive.

``COPY`` is a pure-Python kernel (just allocator → allocator memcpy);
the rest delegate to ``libbonsai_ps.so``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from board.kernels.registry import KernelRegistry
from board.memory.allocator import AllocatorError, TensorAllocator
from board.profiling.timer import Timer
from proto.ops import (
    F_ACTS,
    F_ACTS_OFFSET,
    F_ATTN_FACTOR,
    F_BETA_FAST,
    F_BETA_SLOW,
    F_BIAS,
    F_COLS,
    F_DST,
    F_DST_NB1,
    F_DST_NB2,
    F_DST_NB3,
    F_DST_OFFSET,
    F_DST_TYPE,
    F_ELEMENTS,
    F_EPS,
    F_EXT_FACTOR,
    F_FREQ_BASE,
    F_FREQ_SCALE,
    F_HAS_MASK,
    F_HEAD_DIM,
    F_HEAD_DIM_Q,
    F_HEAD_DIM_V,
    F_INDICES,
    F_INDICES_NB1,
    F_INDICES_NB2,
    F_INDICES_OFFSET,
    F_INDICES_TYPE,
    F_K,
    F_K_NB1,
    F_K_NB2,
    F_K_OFFSET,
    F_K_TENSOR,
    F_LOGIT_SOFTCAP,
    F_MASK,
    F_MASK_NB1,
    F_MASK_OFFSET,
    F_MAX_BIAS,
    F_MODE,
    F_N_CTX_ORIG,
    F_N_DIMS,
    F_N_HEAD,
    F_N_HEAD_KV,
    F_N_KV,
    F_N_TOKEN,
    F_NBYTES,
    F_NE01,
    F_NE02,
    F_NE03,
    F_NE10,
    F_NE11,
    F_NE12,
    F_POSITIONS,
    F_POSITIONS_OFFSET,
    F_Q_NB1,
    F_Q_NB2,
    F_ROWS,
    F_SCALE,
    F_SRC,
    F_SRC0,
    F_SRC0_NB1,
    F_SRC0_NB2,
    F_SRC0_NB3,
    F_SRC0_OFFSET,
    F_SRC0_TYPE,
    F_SRC1,
    F_SRC1_BROADCAST,
    F_SRC1_OFFSET,
    F_SRC_OFFSET,
    F_V_NB1,
    F_V_NB2,
    F_V_OFFSET,
    F_V_TENSOR,
    F_WEIGHTS,
    F_WEIGHTS_OFFSET,
    GOP_ADD_F32,
    GOP_COPY,
    GOP_FLASH_ATTN_EXT_F32,
    GOP_GET_ROWS,
    GOP_MATMUL_Q1A8,
    GOP_MUL_F32,
    GOP_RMS_NORM_F32,
    GOP_ROPE_F32,
    GOP_SCALE_F32,
    GOP_SET_ROWS,
    GOP_SILU_F32,
    GOP_SWIGLU_F32,
    Q1_BLOCK,
)

F32_BYTES = 4
DEFAULT_LIB_PATH = Path(__file__).parent / "libbonsai_ps.so"


def _required(op: dict[str, Any], key: str) -> int:
    if key not in op:
        raise AllocatorError("invalid_request", f"missing {key}")
    value = int(op[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


def _optional_int(op: dict[str, Any], key: str, default: int = 0) -> int:
    if key not in op:
        return default
    value = int(op[key])
    if value < 0:
        raise AllocatorError("invalid_request", f"{key} must be non-negative")
    return value


def _positive(op: dict[str, Any], key: str) -> int:
    value = _required(op, key)
    if value <= 0:
        raise AllocatorError("invalid_request", f"{key} must be positive")
    return value


def _read(allocator: TensorAllocator, handle: int, offset: int, nbytes: int, timer: Timer) -> bytes:
    with timer.section("read"):
        data = allocator.read(handle, offset, nbytes)
    timer.add("bytes_read", nbytes)
    return data


def _write(allocator: TensorAllocator, handle: int, offset: int, data, timer: Timer) -> None:
    with timer.section("write"):
        allocator.write(handle, offset, data)
    timer.add("bytes_written", len(data))


# -- Pure-Python kernel ----------------------------------------------------


class Copy:
    name = GOP_COPY

    def run(self, allocator, op, timer):
        nbytes = _required(op, F_NBYTES)
        data = _read(
            allocator,
            _required(op, "src"),
            _optional_int(op, F_SRC_OFFSET),
            nbytes,
            timer,
        )
        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            data,
            timer,
        )


# -- ctypes-backed kernels -------------------------------------------------


class _NativeKernel:
    """Base helper. Subclasses bind one libbonsai_ps.so entry point."""

    name: str = ""
    symbol: str = ""
    argtypes: tuple = ()

    def __init__(self, lib: ctypes.CDLL):
        fn = getattr(lib, self.symbol)
        fn.argtypes = list(self.argtypes)
        fn.restype = ctypes.c_int
        self._fn = fn

    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"native {self.symbol} failed with rc={rc}")


class MatmulQ1A8(_NativeKernel):
    name = GOP_MATMUL_Q1A8
    symbol = "bonsai_matmul_q1a8"
    argtypes = (
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )

    def run(self, allocator, op, timer):
        from proto import q1a8_layout

        rows = _required(op, F_ROWS)
        cols = _required(op, F_COLS)
        k = _required(op, F_K)
        if k == 0 or k % Q1_BLOCK != 0:
            raise AllocatorError(
                "invalid_request",
                f"{GOP_MATMUL_Q1A8} k must be a positive multiple of {Q1_BLOCK}",
            )

        # Weights are stored in AXIS-packed layout (host backend repacks at
        # upload). Same total bytes as Q1_0 *when rows is a multiple of
        # ROWS_PER_BLOCK*; partial trailing rowblocks pad up to a full block.
        weight_nbytes = q1a8_layout.packed_nbytes(rows, k)
        act_nbytes = cols * k * F32_BYTES
        dst_nbytes = rows * cols * F32_BYTES

        weights = _read(
            allocator,
            _required(op, F_WEIGHTS),
            _optional_int(op, F_WEIGHTS_OFFSET),
            weight_nbytes,
            timer,
        )
        acts = _read(
            allocator,
            _required(op, F_ACTS),
            _optional_int(op, F_ACTS_OFFSET),
            act_nbytes,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(dst_nbytes)
            self._check(
                self._fn(
                    (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights),
                    (ctypes.c_float * (cols * k)).from_buffer_copy(acts),
                    (ctypes.c_float * (rows * cols)).from_buffer(out),
                    ctypes.c_uint32(rows),
                    ctypes.c_uint32(cols),
                    ctypes.c_uint32(k),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class _BinaryF32(_NativeKernel):
    argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )

    def run(self, allocator, op, timer):
        rows = _positive(op, F_ROWS)
        cols = _positive(op, F_COLS)
        broadcast = bool(op.get(F_SRC1_BROADCAST, False))
        elements = rows * cols
        rhs_elements = rows if broadcast else elements

        src0 = _read(
            allocator,
            _required(op, F_SRC0),
            _optional_int(op, F_SRC0_OFFSET),
            elements * F32_BYTES,
            timer,
        )
        src1 = _read(
            allocator,
            _required(op, F_SRC1),
            _optional_int(op, F_SRC1_OFFSET),
            rhs_elements * F32_BYTES,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(elements * F32_BYTES)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(src0),
                    (ctypes.c_float * rhs_elements).from_buffer_copy(src1),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(rows),
                    ctypes.c_uint32(cols),
                    ctypes.c_uint32(1 if broadcast else 0),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class AddF32(_BinaryF32):
    name = GOP_ADD_F32
    symbol = "bonsai_add_f32"


class MulF32(_BinaryF32):
    name = GOP_MUL_F32
    symbol = "bonsai_mul_f32"


class ScaleF32(_NativeKernel):
    name = GOP_SCALE_F32
    symbol = "bonsai_scale_f32"
    argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
    )

    def run(self, allocator, op, timer):
        elements = _positive(op, F_ELEMENTS)
        scale = float(op[F_SCALE]) if F_SCALE in op else 0.0
        if F_SCALE not in op:
            raise AllocatorError("invalid_request", "missing scale")
        bias = float(op.get(F_BIAS, 0.0))

        src = _read(
            allocator,
            _required(op, F_SRC),
            _optional_int(op, F_SRC_OFFSET),
            elements * F32_BYTES,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(elements * F32_BYTES)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(src),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(elements),
                    ctypes.c_float(scale),
                    ctypes.c_float(bias),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class _UnaryF32(_NativeKernel):
    argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    )

    def run(self, allocator, op, timer):
        elements = _positive(op, F_ELEMENTS)
        src = _read(
            allocator,
            _required(op, F_SRC),
            _optional_int(op, F_SRC_OFFSET),
            elements * F32_BYTES,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(elements * F32_BYTES)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(src),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(elements),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class SiluF32(_UnaryF32):
    name = GOP_SILU_F32
    symbol = "bonsai_silu_f32"


class SwigluF32(_NativeKernel):
    name = GOP_SWIGLU_F32
    symbol = "bonsai_swiglu_f32"
    argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    )

    def run(self, allocator, op, timer):
        elements = _positive(op, F_ELEMENTS)
        gate = _read(
            allocator,
            _required(op, F_SRC0),
            _optional_int(op, F_SRC0_OFFSET),
            elements * F32_BYTES,
            timer,
        )
        up = _read(
            allocator,
            _required(op, F_SRC1),
            _optional_int(op, F_SRC1_OFFSET),
            elements * F32_BYTES,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(elements * F32_BYTES)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(gate),
                    (ctypes.c_float * elements).from_buffer_copy(up),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(elements),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class RmsNormF32(_NativeKernel):
    name = GOP_RMS_NORM_F32
    symbol = "bonsai_rms_norm_f32"
    argtypes = (
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
    )

    def run(self, allocator, op, timer):
        rows = _positive(op, F_ROWS)
        cols = _positive(op, F_COLS)
        if F_EPS not in op:
            raise AllocatorError("invalid_request", "missing eps")
        eps = float(op[F_EPS])
        elements = rows * cols

        src = _read(
            allocator,
            _required(op, F_SRC),
            _optional_int(op, F_SRC_OFFSET),
            elements * F32_BYTES,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(elements * F32_BYTES)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(src),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(rows),
                    ctypes.c_uint32(cols),
                    ctypes.c_float(eps),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


class RopeF32(_NativeKernel):
    name = GOP_ROPE_F32
    symbol = "bonsai_rope_f32"
    argtypes = (
        ctypes.POINTER(ctypes.c_float),     # src
        ctypes.POINTER(ctypes.c_int32),     # positions
        ctypes.POINTER(ctypes.c_float),     # dst
        ctypes.c_uint32,                    # head_dim
        ctypes.c_uint32,                    # n_head
        ctypes.c_uint32,                    # n_token
        ctypes.c_uint32,                    # n_dims
        ctypes.c_uint32,                    # mode
        ctypes.c_uint32,                    # n_ctx_orig
        ctypes.c_float,                     # freq_base
        ctypes.c_float,                     # freq_scale
        ctypes.c_float,                     # ext_factor
        ctypes.c_float,                     # attn_factor
        ctypes.c_float,                     # beta_fast
        ctypes.c_float,                     # beta_slow
    )

    def run(self, allocator, op, timer):
        head_dim = _positive(op, F_HEAD_DIM)
        n_head   = _positive(op, F_N_HEAD)
        n_token  = _positive(op, F_N_TOKEN)
        n_dims   = _positive(op, F_N_DIMS)
        mode     = _optional_int(op, F_MODE)
        n_ctx_orig  = _optional_int(op, F_N_CTX_ORIG)
        freq_base   = float(op[F_FREQ_BASE])
        freq_scale  = float(op.get(F_FREQ_SCALE, 1.0))
        ext_factor  = float(op.get(F_EXT_FACTOR, 0.0))
        attn_factor = float(op.get(F_ATTN_FACTOR, 1.0))
        beta_fast   = float(op.get(F_BETA_FAST, 0.0))
        beta_slow   = float(op.get(F_BETA_SLOW, 0.0))

        elements = head_dim * n_head * n_token
        src_nbytes = elements * F32_BYTES
        pos_nbytes = n_token * 4  # int32

        src = _read(
            allocator,
            _required(op, F_SRC),
            _optional_int(op, F_SRC_OFFSET),
            src_nbytes,
            timer,
        )
        positions = _read(
            allocator,
            _required(op, F_POSITIONS),
            _optional_int(op, F_POSITIONS_OFFSET),
            pos_nbytes,
            timer,
        )

        with timer.section("compute"):
            out = bytearray(src_nbytes)
            self._check(
                self._fn(
                    (ctypes.c_float * elements).from_buffer_copy(src),
                    (ctypes.c_int32 * n_token).from_buffer_copy(positions),
                    (ctypes.c_float * elements).from_buffer(out),
                    ctypes.c_uint32(head_dim),
                    ctypes.c_uint32(n_head),
                    ctypes.c_uint32(n_token),
                    ctypes.c_uint32(n_dims),
                    ctypes.c_uint32(mode),
                    ctypes.c_uint32(n_ctx_orig),
                    ctypes.c_float(freq_base),
                    ctypes.c_float(freq_scale),
                    ctypes.c_float(ext_factor),
                    ctypes.c_float(attn_factor),
                    ctypes.c_float(beta_fast),
                    ctypes.c_float(beta_slow),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


# -- KV-cache / FLASH_ATTN kernels ----------------------------------------
#
# These kernels operate on tensors that are too big to copy through a
# bytearray on every call (KV cache, attention K/V views can be hundreds
# of KB). They use ``allocator.slab_pointer`` to get a direct C-side view
# into the slab's user-space CMA mapping and call the C kernel with that
# pointer. Multi-extent fallback copies through a per-kernel scratch
# buffer (same pattern as PLMatmulQ1A8._copy_to_cma_scratch).
#
# Cache coherence is whole-slab on read/write today (slabs.py:PynqSlab),
# so we don't need a separate per-range flush — the slab's flush()
# happens once per write. For pure CPU computation (no DMA) this is
# overkill but correct; the Phase 1.2 range-flush refactor will fix it
# system-wide.


_SLAB_SCRATCH_BUFS: dict[tuple[str, str], object] = {}
_SLAB_SCRATCH_SIZES: dict[tuple[str, str], int] = {}


def _slab_pointer_or_scratch(
    allocator: TensorAllocator,
    handle: int,
    offset: int,
    nbytes: int,
    *,
    kernel: str,
    role: str,
    writeback: bool = False,
) -> tuple[int, object]:
    """Return ``(ctypes-addressable C pointer, owner)`` for a tensor range.

    Fast path: the range lives in a single slab extent → return the slab's
    user-space VA. Returns owner=None.

    Fallback: range spans extents → copy to a persistent CMA scratch buffer
    owned by this ``(kernel, role)`` pair. The caller must call
    ``_scratch_writeback`` after computing if writeback=True (writes the
    scratch back to the slab range). ``owner`` is the scratch buffer in
    that case (kept alive across calls).
    """
    from board.memory.allocator import AllocatorError as _AE
    try:
        ptr = allocator.slab_pointer(handle, offset, nbytes)
        return ptr, None
    except _AE as exc:
        if exc.code != "multi_extent":
            raise

    # Multi-extent: copy through scratch. Caller must writeback.
    key = (kernel, role)
    buf = _SLAB_SCRATCH_BUFS.get(key)
    size = _SLAB_SCRATCH_SIZES.get(key, 0)
    if buf is None or size < nbytes:
        import numpy as np
        from pynq import allocate
        buf = allocate(shape=(nbytes,), dtype=np.uint8)
        size = nbytes
        _SLAB_SCRATCH_BUFS[key] = buf
        _SLAB_SCRATCH_SIZES[key] = size

    # Always fill scratch with current slab contents. This is correct for
    # both read-only (FLASH_ATTN Q/K/V/mask) and read-modify-write (SET_ROWS
    # dst): the kernel may rely on existing data in untouched rows.
    # FLASH_ATTN's dst is fully overwritten so the prefill is wasted there,
    # but multi-extent is the rare path; readability wins over the micro-opt.
    import ctypes as _ct

    import numpy as np
    src = allocator.read(handle, offset, nbytes)
    src_arr = np.frombuffer(src, dtype=np.uint8)
    _ct.memmove(
        buf.ctypes.data_as(_ct.c_void_p),
        src_arr.ctypes.data_as(_ct.c_void_p),
        nbytes,
    )
    return buf.ctypes.data_as(ctypes.c_void_p).value, (buf, handle, offset, nbytes)


def _scratch_writeback(owner, allocator: TensorAllocator) -> None:
    """If a slab_pointer call returned a scratch buffer, write it back."""
    if owner is None:
        return
    buf, handle, offset, nbytes = owner
    allocator.write(handle, offset, bytes(memoryview(buf[:nbytes])))


# -- FLASH_ATTN_EXT_F32 ----------------------------------------------------

# Hard cap on n_kv. Bonsai-1.7B Q1_0 + KV cache must stay within ~296 MiB
# CMA. At n_head_kv=8, head_dim=128 (Bonsai), 28 layers: each KV-cache row
# is 8*128*2=2 KiB; full KV (K+V) at ctx=N is N * 28 * 2 * 2 KiB = N * 112 KiB.
# At n_kv=2048 that's 224 MiB, doesn't fit. Cap at 1024 (= 112 MiB) to stay
# safely within budget; raise if/when the KV layout is tightened.
MAX_FLASH_ATTN_KV = 1024


class FlashAttnExtF32(_NativeKernel):
    name = GOP_FLASH_ATTN_EXT_F32
    symbol = "bonsai_flash_attn_ext_f32"
    argtypes = (
        ctypes.c_void_p,  # q_data
        ctypes.c_size_t, ctypes.c_size_t,            # q_nb1, q_nb2
        ctypes.c_void_p,                              # k_data
        ctypes.c_size_t, ctypes.c_size_t,            # k_nb1, k_nb2
        ctypes.c_void_p,                              # v_data
        ctypes.c_size_t, ctypes.c_size_t,            # v_nb1, v_nb2
        ctypes.c_void_p,                              # mask_data (may be NULL)
        ctypes.c_size_t,                              # mask_nb1
        ctypes.c_void_p,                              # dst
        ctypes.c_size_t, ctypes.c_size_t,            # dst_nb1, dst_nb2
        ctypes.c_uint32, ctypes.c_uint32,            # head_dim_q, head_dim_v
        ctypes.c_uint32, ctypes.c_uint32,            # n_head, n_head_kv
        ctypes.c_uint32, ctypes.c_uint32,            # n_kv, n_token
        ctypes.c_float,                               # scale
    )

    def run(self, allocator, op, timer):
        head_dim_q = _positive(op, F_HEAD_DIM_Q)
        head_dim_v = _positive(op, F_HEAD_DIM_V)
        n_head = _positive(op, F_N_HEAD)
        n_head_kv = _positive(op, F_N_HEAD_KV)
        n_kv = _positive(op, F_N_KV)
        n_token = _positive(op, F_N_TOKEN)
        scale = float(op.get(F_SCALE, 1.0))
        max_bias = float(op.get(F_MAX_BIAS, 0.0))
        softcap = float(op.get(F_LOGIT_SOFTCAP, 0.0))
        if max_bias != 0.0 or softcap != 0.0:
            raise AllocatorError(
                "invalid_request",
                "FLASH_ATTN_EXT_F32 doesn't support ALiBi or softcap",
            )
        if n_kv > MAX_FLASH_ATTN_KV:
            raise AllocatorError(
                "invalid_request",
                f"FLASH_ATTN_EXT_F32 n_kv={n_kv} exceeds runtime cap "
                f"({MAX_FLASH_ATTN_KV}); raise MAX_FLASH_ATTN_KV or shrink ctx",
            )
        if n_head % n_head_kv != 0:
            raise AllocatorError(
                "invalid_request", "n_head must be a multiple of n_head_kv"
            )

        q_nb1 = int(op[F_Q_NB1])
        q_nb2 = int(op[F_Q_NB2])
        k_nb1 = int(op[F_K_NB1])
        k_nb2 = int(op[F_K_NB2])
        v_nb1 = int(op[F_V_NB1])
        v_nb2 = int(op[F_V_NB2])
        dst_nb1 = int(op[F_DST_NB1])
        dst_nb2 = int(op[F_DST_NB2])
        has_mask = bool(op.get(F_HAS_MASK, False))
        mask_nb1 = int(op.get(F_MASK_NB1, 0))

        # Tensor sizes — used both for slab-pointer validation and for the
        # multi-extent scratch fallback. Use the largest extent we'll touch.
        q_nbytes = q_nb2 * n_head if q_nb2 else q_nb1 * n_token
        k_nbytes = k_nb2 * n_head_kv if k_nb2 else k_nb1 * n_kv
        v_nbytes = v_nb2 * n_head_kv if v_nb2 else v_nb1 * n_kv
        mask_nbytes = mask_nb1 * n_token if has_mask else 0
        dst_nbytes = dst_nb2 * n_token if dst_nb2 else dst_nb1 * n_head

        with timer.section("read"):
            q_ptr, q_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_SRC0),
                _optional_int(op, F_SRC0_OFFSET), q_nbytes,
                kernel="flash_attn", role="q",
            )
            k_ptr, k_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_K_TENSOR),
                _optional_int(op, F_K_OFFSET), k_nbytes,
                kernel="flash_attn", role="k",
            )
            v_ptr, v_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_V_TENSOR),
                _optional_int(op, F_V_OFFSET), v_nbytes,
                kernel="flash_attn", role="v",
            )
            mask_ptr = 0
            mask_own = None
            if has_mask:
                mask_ptr, mask_own = _slab_pointer_or_scratch(
                    allocator, _required(op, F_MASK),
                    _optional_int(op, F_MASK_OFFSET), mask_nbytes,
                    kernel="flash_attn", role="mask",
                )

            # dst is write-only; we allocate scratch only if it spans extents.
            dst_ptr, dst_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_DST),
                _optional_int(op, F_DST_OFFSET), dst_nbytes,
                kernel="flash_attn", role="dst", writeback=True,
            )

        timer.add("bytes_read", q_nbytes + k_nbytes + v_nbytes + mask_nbytes)

        with timer.section("compute"):
            self._check(
                self._fn(
                    ctypes.c_void_p(q_ptr),
                    ctypes.c_size_t(q_nb1), ctypes.c_size_t(q_nb2),
                    ctypes.c_void_p(k_ptr),
                    ctypes.c_size_t(k_nb1), ctypes.c_size_t(k_nb2),
                    ctypes.c_void_p(v_ptr),
                    ctypes.c_size_t(v_nb1), ctypes.c_size_t(v_nb2),
                    ctypes.c_void_p(mask_ptr) if mask_ptr else ctypes.c_void_p(0),
                    ctypes.c_size_t(mask_nb1),
                    ctypes.c_void_p(dst_ptr),
                    ctypes.c_size_t(dst_nb1), ctypes.c_size_t(dst_nb2),
                    ctypes.c_uint32(head_dim_q), ctypes.c_uint32(head_dim_v),
                    ctypes.c_uint32(n_head), ctypes.c_uint32(n_head_kv),
                    ctypes.c_uint32(n_kv), ctypes.c_uint32(n_token),
                    ctypes.c_float(scale),
                )
            )

        with timer.section("write"):
            _scratch_writeback(dst_own, allocator)
        timer.add("bytes_written", dst_nbytes)


# -- GET_ROWS --------------------------------------------------------------


class GetRows:
    """GET_ROWS for src0 type ∈ {f32, f16, q1_0}, indices type i32, dst f32.

    Dispatches to one of three C entry points based on the src0_type tag.
    The Python class is intentionally a single registered op (named
    GOP_GET_ROWS) — the dispatch happens inside .run().
    """
    name = GOP_GET_ROWS

    def __init__(self, lib: ctypes.CDLL):
        common_argtypes = (
            ctypes.c_void_p,                              # src0
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # src0_nb1/2/3
            ctypes.c_void_p,                              # indices
            ctypes.c_size_t, ctypes.c_size_t,            # indices_nb1/2
            ctypes.c_void_p,                              # dst
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # dst_nb1/2/3
            ctypes.c_uint32, ctypes.c_uint32,            # head_dim, ne01
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # ne10/11/12
        )
        self._fns = {}
        for tag, sym in (
            ("f32",  "bonsai_get_rows_f32"),
            ("f16",  "bonsai_get_rows_f16"),
            ("q1_0", "bonsai_get_rows_q1_0"),
        ):
            fn = getattr(lib, sym)
            fn.argtypes = list(common_argtypes)
            fn.restype = ctypes.c_int
            self._fns[tag] = fn

    def run(self, allocator, op, timer):
        src0_type = str(op.get(F_SRC0_TYPE, ""))
        if src0_type not in self._fns:
            raise AllocatorError(
                "invalid_request",
                f"GET_ROWS src0_type={src0_type!r} not supported (need f32/f16/q1_0)",
            )
        if str(op.get(F_INDICES_TYPE, "i32")) != "i32":
            raise AllocatorError(
                "invalid_request", "GET_ROWS only supports i32 indices",
            )

        head_dim = _positive(op, F_HEAD_DIM)
        ne01 = _positive(op, F_NE01)
        ne10 = _positive(op, F_NE10)
        ne11 = max(1, int(op.get(F_NE11, 1)))
        ne12 = max(1, int(op.get(F_NE12, 1)))

        src0_nb1 = int(op[F_SRC0_NB1])
        src0_nb2 = int(op.get(F_SRC0_NB2, src0_nb1 * ne01))
        src0_nb3 = int(op.get(F_SRC0_NB3, src0_nb2))
        indices_nb1 = int(op.get(F_INDICES_NB1, ne10 * 4))
        indices_nb2 = int(op.get(F_INDICES_NB2, indices_nb1 * ne11))
        dst_nb1 = int(op[F_DST_NB1])
        dst_nb2 = int(op.get(F_DST_NB2, dst_nb1 * ne10))
        dst_nb3 = int(op.get(F_DST_NB3, dst_nb2))

        src0_handle = _required(op, F_SRC0)
        src0_base_offset = _optional_int(op, F_SRC0_OFFSET)
        dst_handle = _required(op, F_DST)
        dst_base_offset = _optional_int(op, F_DST_OFFSET)
        indices_handle = _required(op, F_INDICES)
        indices_offset = _optional_int(op, F_INDICES_OFFSET)
        indices_nbytes = indices_nb2 * ne12

        # The naive "slab_pointer the whole src0 table" path falls into the
        # multi-extent scratch copy for any tensor bigger than one CMA slab
        # (~32 MiB). The Q1_0 token embedding (~43 MiB for Bonsai) hits this
        # path on EVERY call and dominates wall time.
        #
        # Fix: read the indices upfront (small), then iterate ne12*ne11*ne10
        # output rows. For each, resolve the slab_pointer for just the source
        # row (F32/F16) or packed rowblock (Q1_0) and just the dst row, then
        # call the C kernel with ne10=ne11=ne12=1. Python overhead is
        # ~n_indices x ~20us, negligible for normal embedding lookups.
        zero_idx = (ctypes.c_int32 * 1)(0)
        zero_idx_ptr = ctypes.cast(zero_idx, ctypes.c_void_p)
        fn = self._fns[src0_type]
        q1a8_layout = None
        rowblock_nbytes = 0
        if src0_type == "q1_0":
            from proto import q1a8_layout as _q1a8_layout

            q1a8_layout = _q1a8_layout
            rowblock_nbytes = q1a8_layout.packed_bytes_per_rowblock(head_dim)

        with timer.section("read"):
            indices_bytes = allocator.read(indices_handle, indices_offset, indices_nbytes)
        timer.add("bytes_read", indices_nbytes)
        n_indices = ne10 * ne11 * ne12

        compute_bytes_read = 0
        with timer.section("compute"):
            for i12 in range(ne12):
                for i11 in range(ne11):
                    for i10 in range(ne10):
                        idx_byte_off = i10 * 4 + i11 * indices_nb1 + i12 * indices_nb2
                        i01 = int.from_bytes(
                            indices_bytes[idx_byte_off:idx_byte_off + 4],
                            "little", signed=True,
                        )
                        if i01 < 0 or i01 >= ne01:
                            raise RuntimeError(
                                f"GET_ROWS index out of range: {i01} (ne01={ne01})"
                            )
                        if src0_type == "q1_0":
                            assert q1a8_layout is not None
                            rowblock = i01 // q1a8_layout.ROWS_PER_BLOCK
                            lane_idx = (ctypes.c_int32 * 1)(i01 % q1a8_layout.ROWS_PER_BLOCK)
                            idx_ptr = ctypes.cast(lane_idx, ctypes.c_void_p)
                            src_row_offset = (
                                src0_base_offset
                                + rowblock * rowblock_nbytes
                                + i11 * src0_nb2
                                + i12 * src0_nb3
                            )
                            src_nbytes = rowblock_nbytes
                            c_ne01 = q1a8_layout.ROWS_PER_BLOCK
                        else:
                            idx_ptr = zero_idx_ptr
                            src_row_offset = (
                                src0_base_offset
                                + i01 * src0_nb1
                                + i11 * src0_nb2
                                + i12 * src0_nb3
                            )
                            src_nbytes = src0_nb1
                            c_ne01 = 1
                        dst_row_offset = (
                            dst_base_offset
                            + i10 * dst_nb1
                            + i11 * dst_nb2
                            + i12 * dst_nb3
                        )
                        src_buf = None
                        dst_buf = None
                        try:
                            # Each row/rowblock is single-extent on real PYNQ slabs.
                            src_ptr = allocator.slab_pointer(
                                src0_handle, src_row_offset, src_nbytes,
                            )
                        except AllocatorError as exc:
                            if exc.code != "invalid_request":
                                raise
                            src_buf = ctypes.create_string_buffer(src_nbytes)
                            src_bytes = allocator.read(src0_handle, src_row_offset, src_nbytes)
                            ctypes.memmove(src_buf, src_bytes, src_nbytes)
                            src_ptr = ctypes.cast(src_buf, ctypes.c_void_p).value
                        try:
                            dst_ptr = allocator.slab_pointer(
                                dst_handle, dst_row_offset, dst_nb1,
                            )
                        except AllocatorError as exc:
                            if exc.code != "invalid_request":
                                raise
                            dst_buf = ctypes.create_string_buffer(dst_nb1)
                            dst_ptr = ctypes.cast(dst_buf, ctypes.c_void_p).value
                        compute_bytes_read += src_nbytes
                        # Call C for exactly one output row. F32/F16 receive
                        # a pointer to the selected source row and index 0;
                        # Q1_0 receives the packed rowblock and lane index.
                        rc = fn(
                            ctypes.c_void_p(src_ptr),
                            ctypes.c_size_t(src0_nb1),
                            ctypes.c_size_t(src0_nb1),
                            ctypes.c_size_t(src0_nb1),
                            idx_ptr,
                            ctypes.c_size_t(4), ctypes.c_size_t(4),
                            ctypes.c_void_p(dst_ptr),
                            ctypes.c_size_t(dst_nb1),
                            ctypes.c_size_t(dst_nb1),
                            ctypes.c_size_t(dst_nb1),
                            ctypes.c_uint32(head_dim), ctypes.c_uint32(c_ne01),
                            ctypes.c_uint32(1), ctypes.c_uint32(1), ctypes.c_uint32(1),
                        )
                        if rc != 0:
                            raise RuntimeError(
                                f"bonsai_get_rows_{src0_type} rc={rc}"
                            )
                        if dst_buf is not None:
                            allocator.write(
                                dst_handle, dst_row_offset,
                                memoryview(dst_buf.raw)[:dst_nb1],
                            )

        timer.add("bytes_read", compute_bytes_read)
        # On real PYNQ slabs, slab_pointer writes already landed in the dst
        # slab's CMA mapping. The fake-slab test fallback writes through
        # allocator.write row-by-row above.
        timer.add("bytes_written", n_indices * dst_nb1)


# -- SET_ROWS --------------------------------------------------------------


class SetRows:
    """SET_ROWS src0 F32 → dst F16. Indices i32 or i64."""
    name = GOP_SET_ROWS

    def __init__(self, lib: ctypes.CDLL):
        common_argtypes = (
            ctypes.c_void_p,                              # src0
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # src0_nb1/2/3
            ctypes.c_void_p,                              # indices
            ctypes.c_size_t, ctypes.c_size_t,            # indices_nb1/2
            ctypes.c_void_p,                              # dst
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # dst_nb1/2/3
            ctypes.c_uint32,                              # head_dim
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,  # ne01/02/03
            ctypes.c_uint32, ctypes.c_uint32,            # ne11/12
        )
        self._fns = {}
        for tag, sym in (
            ("i32", "bonsai_set_rows_f32_to_f16_i32"),
            ("i64", "bonsai_set_rows_f32_to_f16_i64"),
        ):
            fn = getattr(lib, sym)
            fn.argtypes = list(common_argtypes)
            fn.restype = ctypes.c_int
            self._fns[tag] = fn

    def run(self, allocator, op, timer):
        if str(op.get(F_DST_TYPE, "f16")) != "f16":
            raise AllocatorError("invalid_request", "SET_ROWS dst must be f16")
        idx_type = str(op.get(F_INDICES_TYPE, "i32"))
        if idx_type not in self._fns:
            raise AllocatorError(
                "invalid_request",
                f"SET_ROWS indices type {idx_type!r} unsupported (i32/i64 only)",
            )

        head_dim = _positive(op, F_HEAD_DIM)
        ne01 = _positive(op, F_NE01)
        ne02 = int(op.get(F_NE02, 1)) or 1
        ne03 = int(op.get(F_NE03, 1)) or 1
        ne11 = int(op.get(F_NE11, 1)) or 1
        ne12 = int(op.get(F_NE12, 1)) or 1

        src0_nb1 = int(op[F_SRC0_NB1])
        src0_nb2 = int(op.get(F_SRC0_NB2, src0_nb1 * ne01))
        src0_nb3 = int(op.get(F_SRC0_NB3, src0_nb2 * ne02))
        idx_elem = 4 if idx_type == "i32" else 8
        indices_nb1 = int(op.get(F_INDICES_NB1, ne01 * idx_elem))
        indices_nb2 = int(op.get(F_INDICES_NB2, indices_nb1 * ne11))
        dst_nb1 = int(op[F_DST_NB1])
        dst_nb2 = int(op.get(F_DST_NB2, dst_nb1))
        dst_nb3 = int(op.get(F_DST_NB3, dst_nb2))

        src0_nbytes = src0_nb3 * ne03
        indices_nbytes = indices_nb2 * ne12
        # dst is the KV cache — large. We touch [0, max_row*dst_nb1) potentially.
        # For correctness we just claim the whole tensor allocation.
        dst_handle = _required(op, F_DST)
        dst_record = allocator.describe(dst_handle)
        dst_nbytes = int(dst_record["nbytes"]) - _optional_int(op, F_DST_OFFSET)

        with timer.section("read"):
            src_ptr, src_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_SRC0),
                _optional_int(op, F_SRC0_OFFSET), src0_nbytes,
                kernel="set_rows", role="src0",
            )
            idx_ptr, idx_own = _slab_pointer_or_scratch(
                allocator, _required(op, F_INDICES),
                _optional_int(op, F_INDICES_OFFSET), indices_nbytes,
                kernel="set_rows", role="indices",
            )
            dst_ptr, dst_own = _slab_pointer_or_scratch(
                allocator, dst_handle,
                _optional_int(op, F_DST_OFFSET), dst_nbytes,
                kernel="set_rows", role="dst", writeback=True,
            )
        timer.add("bytes_read", src0_nbytes + indices_nbytes)

        fn = self._fns[idx_type]
        with timer.section("compute"):
            rc = fn(
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(src0_nb1), ctypes.c_size_t(src0_nb2), ctypes.c_size_t(src0_nb3),
                ctypes.c_void_p(idx_ptr),
                ctypes.c_size_t(indices_nb1), ctypes.c_size_t(indices_nb2),
                ctypes.c_void_p(dst_ptr),
                ctypes.c_size_t(dst_nb1), ctypes.c_size_t(dst_nb2), ctypes.c_size_t(dst_nb3),
                ctypes.c_uint32(head_dim),
                ctypes.c_uint32(ne01), ctypes.c_uint32(ne02), ctypes.c_uint32(ne03),
                ctypes.c_uint32(ne11), ctypes.c_uint32(ne12),
            )
            if rc != 0:
                raise RuntimeError(f"bonsai_set_rows_f32_to_f16_{idx_type} rc={rc}")

        # SET_ROWS' "dst" really is the same tensor logically as the KV
        # cache — writes accumulate. Only the changed rows need writeback,
        # but for the multi-extent path we wrote into scratch, so flush
        # the whole scratch range. (When slab_pointer succeeds, dst_own
        # is None and this is a no-op.)
        with timer.section("write"):
            _scratch_writeback(dst_own, allocator)
        timer.add("bytes_written", ne01 * head_dim * 2)  # F16 bytes


# -- registry wiring -------------------------------------------------------


NATIVE_KERNELS = (
    MatmulQ1A8,
    AddF32,
    MulF32,
    ScaleF32,
    SiluF32,
    SwigluF32,
    RmsNormF32,
    RopeF32,
    FlashAttnExtF32,
    # GetRows / SetRows are constructed below — they need lib explicitly,
    # not the _NativeKernel(lib) pattern, because they bind multiple C
    # functions each.
)


def load_lib(path: Path | None = None) -> ctypes.CDLL:
    """Load ``libbonsai_ps.so`` from ``path`` or the canonical location."""
    actual = Path(path) if path else DEFAULT_LIB_PATH
    if not actual.exists():
        raise FileNotFoundError(
            f"native kernel library missing: {actual}. "
            f"Run `make -C board/kernels/ps` to build it."
        )
    return ctypes.CDLL(str(actual))


def register_all(registry: KernelRegistry, lib_path: Path | None = None) -> None:
    """Register the pure-Python COPY kernel and every libbonsai_ps.so kernel."""
    registry.register(Copy())
    lib = load_lib(lib_path)
    for kls in NATIVE_KERNELS:
        registry.register(kls(lib))
    # GetRows / SetRows bind multiple C symbols each; construct directly.
    registry.register(GetRows(lib))
    registry.register(SetRows(lib))
