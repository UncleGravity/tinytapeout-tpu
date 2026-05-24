"""Shared fixtures for board-side unit tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from board.kernels.ps import native as ps_native
from board.kernels.registry import KernelRegistry
from board.memory.allocator import TensorAllocator
from board.memory.slabs import fake_slabs

PS_DIR = Path(__file__).resolve().parents[1] / "kernels" / "ps"


@pytest.fixture(scope="session")
def native_lib_path(tmp_path_factory) -> Path:
    """Build libbonsai_ps.so into a session tmpdir, never into the source tree."""
    out_dir = tmp_path_factory.mktemp("ps_native")
    subprocess.run(
        ["make", "-C", str(PS_DIR), f"OUT_DIR={out_dir}"],
        check=True,
    )
    return out_dir / "libbonsai_ps.so"


@pytest.fixture
def allocator() -> TensorAllocator:
    return TensorAllocator(fake_slabs(total_bytes=1 << 20, slab_bytes=256 * 1024))


@pytest.fixture
def registry(native_lib_path: Path) -> KernelRegistry:
    reg = KernelRegistry()
    ps_native.register_all(reg, lib_path=native_lib_path)
    return reg
