"""Deterministic test inputs reused across kernel and dispatch tests."""

from __future__ import annotations

import struct


def f32_bytes(n: int, seed: int = 0) -> bytes:
    """Return ``n`` deterministic F32 values packed little-endian."""
    return struct.pack(
        f"<{n}f",
        *(((index + seed) * 0.0173 - 1.5) for index in range(n)),
    )


def deterministic_bytes(nbytes: int) -> bytes:
    return bytes((index * 17 + 3) % 251 for index in range(nbytes))
