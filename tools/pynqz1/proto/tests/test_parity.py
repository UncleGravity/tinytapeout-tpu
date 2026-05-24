"""Assert proto/ops.py and proto/ops.h declare the same constants."""

from __future__ import annotations

import re
from pathlib import Path

from proto import ops


HEADER = Path(__file__).resolve().parents[1] / "ops.h"

_INT_DECL = re.compile(
    r"inline\s+constexpr\s+(?:int|std::uint16_t)\s+(\w+)\s*=\s*([0-9]+)\s*;"
)
_STR_DECL = re.compile(
    r'inline\s+constexpr\s+const\s+char\s+(\w+)\s*\[\s*\]\s*=\s*"([^"]*)"\s*;'
)


def _parse_header() -> dict[str, object]:
    text = HEADER.read_text()
    values: dict[str, object] = {}
    for match in _INT_DECL.finditer(text):
        values[match.group(1)] = int(match.group(2))
    for match in _STR_DECL.finditer(text):
        values[match.group(1)] = match.group(2)
    return values


def test_header_and_python_agree():
    header = _parse_header()
    python = {
        name: getattr(ops, name)
        for name in dir(ops)
        if not name.startswith("_")
        and isinstance(getattr(ops, name), (str, int))
        and name.isupper()
    }

    missing_in_header = sorted(set(python) - set(header))
    missing_in_python = sorted(set(header) - set(python))
    mismatched = sorted(
        name for name in set(python) & set(header) if python[name] != header[name]
    )

    assert not missing_in_header, f"in ops.py but not ops.h: {missing_in_header}"
    assert not missing_in_python, f"in ops.h but not ops.py: {missing_in_python}"
    assert not mismatched, (
        "value mismatch: "
        + ", ".join(f"{n} py={python[n]!r} h={header[n]!r}" for n in mismatched)
    )
