from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Any


class NativeKernels:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.load_error: str | None = None
        self._lib: ctypes.CDLL | None = None
        self._matmul_q1a8 = None
        self._add_f32 = None
        self._mul_f32 = None
        self._scale_f32 = None
        self._silu_f32 = None
        self._swiglu_f32 = None
        self._rms_norm_f32 = None
        self._load()

    @property
    def available(self) -> bool:
        return self._matmul_q1a8 is not None

    def matmul_q1a8(
        self,
        weights: bytes,
        acts: bytes,
        output: bytearray,
        rows: int,
        cols: int,
        k: int,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        if self._matmul_q1a8 is None:
            return False

        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        weight_array = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        act_array = (ctypes.c_float * (cols * k)).from_buffer_copy(acts)
        output_array = (ctypes.c_float * (rows * cols)).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = self._matmul_q1a8(
            weight_array,
            act_array,
            output_array,
            ctypes.c_uint32(rows),
            ctypes.c_uint32(cols),
            ctypes.c_uint32(k),
        )
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native MATMUL_Q1A8 failed with rc={rc}")
        return True

    def binary_f32(
        self,
        op: str,
        src0: bytes,
        src1: bytes,
        output: bytearray,
        rows: int,
        cols: int,
        src1_broadcast: bool,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        fn = self._add_f32 if op == "add" else self._mul_f32 if op == "mul" else None
        if fn is None:
            return False

        elements = rows * cols
        rhs_elements = rows if src1_broadcast else elements
        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        src0_array = (ctypes.c_float * elements).from_buffer_copy(src0)
        src1_array = (ctypes.c_float * rhs_elements).from_buffer_copy(src1)
        output_array = (ctypes.c_float * elements).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = fn(
            src0_array,
            src1_array,
            output_array,
            ctypes.c_uint32(rows),
            ctypes.c_uint32(cols),
            ctypes.c_uint32(1 if src1_broadcast else 0),
        )
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native {op.upper()}_F32 failed with rc={rc}")
        return True

    def scale_f32(
        self,
        src: bytes,
        output: bytearray,
        elements: int,
        scale: float,
        bias: float,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        if self._scale_f32 is None:
            return False

        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        src_array = (ctypes.c_float * elements).from_buffer_copy(src)
        output_array = (ctypes.c_float * elements).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = self._scale_f32(
            src_array,
            output_array,
            ctypes.c_uint32(elements),
            ctypes.c_float(scale),
            ctypes.c_float(bias),
        )
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native SCALE_F32 failed with rc={rc}")
        return True

    def silu_f32(
        self,
        src: bytes,
        output: bytearray,
        elements: int,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        if self._silu_f32 is None:
            return False

        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        src_array = (ctypes.c_float * elements).from_buffer_copy(src)
        output_array = (ctypes.c_float * elements).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = self._silu_f32(src_array, output_array, ctypes.c_uint32(elements))
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native SILU_F32 failed with rc={rc}")
        return True

    def swiglu_f32(
        self,
        gate: bytes,
        up: bytes,
        output: bytearray,
        elements: int,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        if self._swiglu_f32 is None:
            return False

        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        gate_array = (ctypes.c_float * elements).from_buffer_copy(gate)
        up_array = (ctypes.c_float * elements).from_buffer_copy(up)
        output_array = (ctypes.c_float * elements).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = self._swiglu_f32(gate_array, up_array, output_array, ctypes.c_uint32(elements))
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native SWIGLU_F32 failed with rc={rc}")
        return True

    def rms_norm_f32(
        self,
        src: bytes,
        output: bytearray,
        rows: int,
        cols: int,
        eps: float,
        profile: dict[str, Any] | None = None,
    ) -> bool:
        if self._rms_norm_f32 is None:
            return False

        elements = rows * cols
        marshal_start_ns = time.perf_counter_ns() if profile is not None else 0
        src_array = (ctypes.c_float * elements).from_buffer_copy(src)
        output_array = (ctypes.c_float * elements).from_buffer(output)
        if profile is not None:
            _add_elapsed_us(profile, "native_marshal_us", marshal_start_ns)

        kernel_start_ns = time.perf_counter_ns() if profile is not None else 0
        rc = self._rms_norm_f32(
            src_array,
            output_array,
            ctypes.c_uint32(rows),
            ctypes.c_uint32(cols),
            ctypes.c_float(eps),
        )
        if profile is not None:
            _add_elapsed_us(profile, "native_kernel_us", kernel_start_ns)
        if rc != 0:
            raise RuntimeError(f"bonsai native RMS_NORM_F32 failed with rc={rc}")
        return True

    def _load(self) -> None:
        env_path = os.environ.get("PYNQ_PS_LIB")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path(__file__).resolve().parent / "native" / "libbonsai_ps.so")

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                lib = ctypes.CDLL(str(candidate))
                self._matmul_q1a8 = _optional_fn(
                    lib,
                    "bonsai_matmul_q1a8",
                    [
                        ctypes.POINTER(ctypes.c_uint8),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                    ],
                )
                self._add_f32 = _optional_fn(lib, "bonsai_add_f32", _binary_f32_args())
                self._mul_f32 = _optional_fn(lib, "bonsai_mul_f32", _binary_f32_args())
                self._scale_f32 = _optional_fn(
                    lib,
                    "bonsai_scale_f32",
                    [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_uint32,
                        ctypes.c_float,
                        ctypes.c_float,
                    ],
                )
                self._silu_f32 = _optional_fn(
                    lib,
                    "bonsai_silu_f32",
                    [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_uint32,
                    ],
                )
                self._swiglu_f32 = _optional_fn(
                    lib,
                    "bonsai_swiglu_f32",
                    [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_uint32,
                    ],
                )
                self._rms_norm_f32 = _optional_fn(
                    lib,
                    "bonsai_rms_norm_f32",
                    [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_uint32,
                        ctypes.c_uint32,
                        ctypes.c_float,
                    ],
                )
                if self._matmul_q1a8 is None:
                    continue
                self.path = candidate
                self._lib = lib
                return
            except OSError as exc:
                self.load_error = str(exc)

        if env_path and self._matmul_q1a8 is None:
            print(
                f"bonsaid: could not load PYNQ_PS_LIB={env_path}: "
                f"{self.load_error or 'file not found'}",
                file=sys.stderr,
                flush=True,
            )


def _optional_fn(lib: ctypes.CDLL, name: str, argtypes: list[object]):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        return None

    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


def _binary_f32_args() -> list[object]:
    return [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]


def _add_elapsed_us(profile: dict[str, Any], key: str, start_ns: int) -> None:
    profile[key] = int(profile.get(key, 0)) + max(
        0,
        (time.perf_counter_ns() - start_ns) // 1000,
    )


_native_kernels: NativeKernels | None = None


def get_native_kernels() -> NativeKernels:
    global _native_kernels
    if _native_kernels is None:
        _native_kernels = NativeKernels()
    return _native_kernels
