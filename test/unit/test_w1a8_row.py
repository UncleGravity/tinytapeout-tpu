from pathlib import Path

from cocotb_tools.runner import get_runner

from helpers import RTL_DIR

HERE = Path(__file__).parent


def test_w1a8_row():
    build_dir = HERE / "sim_build" / "w1a8_row"
    runner = get_runner("icarus")
    runner.build(
        sources=[RTL_DIR / "w1a8_pe.v", RTL_DIR / "w1a8_row.v"],
        hdl_toplevel="w1a8_row",
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        build_args=["-g2012"],
    )
    runner.test(
        hdl_toplevel="w1a8_row",
        test_module="w1a8_row_cocotb",
        test_dir=HERE,
        build_dir=build_dir,
    )
