from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


class NativeKernels:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.load_error: str | None = None
        self._lib: ctypes.CDLL | None = None
        self._matmul_q1a8 = None
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
    ) -> bool:
        if self._matmul_q1a8 is None:
            return False

        weight_array = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        act_array = (ctypes.c_float * (cols * k)).from_buffer_copy(acts)
        output_array = (ctypes.c_float * (rows * cols)).from_buffer(output)

        rc = self._matmul_q1a8(
            weight_array,
            act_array,
            output_array,
            ctypes.c_uint32(rows),
            ctypes.c_uint32(cols),
            ctypes.c_uint32(k),
        )
        if rc != 0:
            raise RuntimeError(f"bonsai native MATMUL_Q1A8 failed with rc={rc}")
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
                fn = lib.bonsai_matmul_q1a8
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                ]
                fn.restype = ctypes.c_int
                self.path = candidate
                self._lib = lib
                self._matmul_q1a8 = fn
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


_native_kernels: NativeKernels | None = None


def get_native_kernels() -> NativeKernels:
    global _native_kernels
    if _native_kernels is None:
        _native_kernels = NativeKernels()
    return _native_kernels
