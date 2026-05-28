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
    F_BIAS,
    F_COLS,
    F_DST,
    F_DST_OFFSET,
    F_ELEMENTS,
    F_EPS,
    F_FREQ_BASE,
    F_FREQ_SCALE,
    F_HEAD_DIM,
    F_K,
    F_MODE,
    F_N_DIMS,
    F_N_HEAD,
    F_N_TOKEN,
    F_NBYTES,
    F_POSITIONS,
    F_POSITIONS_OFFSET,
    F_ROWS,
    F_SCALE,
    F_SRC,
    F_SRC0,
    F_SRC0_OFFSET,
    F_SRC1,
    F_SRC1_BROADCAST,
    F_SRC1_OFFSET,
    F_SRC_OFFSET,
    F_WEIGHTS,
    F_WEIGHTS_OFFSET,
    GOP_ADD_F32,
    GOP_COPY,
    GOP_MATMUL_Q1A8,
    GOP_MUL_F32,
    GOP_RMS_NORM_F32,
    GOP_ROPE_F32,
    GOP_SCALE_F32,
    GOP_SILU_F32,
    GOP_SWIGLU_F32,
    Q1_BLOCK,
    Q1_BLOCK_BYTES,
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
        ctypes.c_float,                     # freq_base
        ctypes.c_float,                     # freq_scale
    )

    def run(self, allocator, op, timer):
        head_dim = _positive(op, F_HEAD_DIM)
        n_head   = _positive(op, F_N_HEAD)
        n_token  = _positive(op, F_N_TOKEN)
        n_dims   = _positive(op, F_N_DIMS)
        mode     = _optional_int(op, F_MODE)
        freq_base  = float(op[F_FREQ_BASE])
        freq_scale = float(op.get(F_FREQ_SCALE, 1.0))

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
                    ctypes.c_float(freq_base),
                    ctypes.c_float(freq_scale),
                )
            )

        _write(
            allocator,
            _required(op, F_DST),
            _optional_int(op, F_DST_OFFSET),
            out,
            timer,
        )


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
