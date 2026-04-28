"""Per-module cocotb unit tests, dispatched by pytest.

Each entry below names an RTL toplevel, the cocotb test module that drives
it, and the RTL sources to compile. Adding a new unit test is a one-line
addition; the actual @cocotb.test() functions live in the matching
*_cocotb.py file (cocotb reimports it inside the simulator process).
"""
from pathlib import Path

import pytest
from cocotb_tools.runner import get_results, get_runner


HERE = Path(__file__).parent
RTL_DIR = HERE.resolve().parents[1] / "src/rtl"


UNIT_TESTS = [
    ("w1a8_pe",    "w1a8_pe_cocotb",    ["w1a8_pe.v"]),
    ("w1a8_row",   "w1a8_row_cocotb",   ["w1a8_pe.v", "w1a8_row.v"]),
    ("w1a8_array", "w1a8_array_cocotb", ["w1a8_pe.v", "w1a8_row.v", "w1a8_array.v"]),
]


@pytest.mark.parametrize(
    ("toplevel", "cocotb_module", "sources"),
    UNIT_TESTS,
    ids=[t[0] for t in UNIT_TESTS],
)
def test_module(toplevel, cocotb_module, sources):
    build_dir = HERE / "sim_build" / toplevel
    runner = get_runner("icarus")
    runner.build(
        sources=[RTL_DIR / s for s in sources],
        hdl_toplevel=toplevel,
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        build_args=["-g2012"],
    )
    results = runner.test(
        hdl_toplevel=toplevel,
        test_module=cocotb_module,
        test_dir=HERE,
        build_dir=build_dir,
    )
    n_tests, n_fails = get_results(results)
    assert n_fails == 0, f"{n_fails}/{n_tests} cocotb test(s) failed (see {results})"
