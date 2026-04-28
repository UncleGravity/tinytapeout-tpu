from pathlib import Path

from cocotb_tools.runner import get_runner

from helpers import RTL_DIR

HERE = Path(__file__).parent


def test_tt_top():
    build_dir = HERE / "sim_build" / "tt_top"
    runner = get_runner("icarus")
    runner.build(
        sources=[
            RTL_DIR / "w1a8_pe.v",
            RTL_DIR / "w1a8_row.v",
            RTL_DIR / "w1a8_array.v",
            RTL_DIR / "project.v",
        ],
        hdl_toplevel="tt_um_unclegravity_tpu",
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        build_args=["-g2012"],
    )
    runner.test(
        hdl_toplevel="tt_um_unclegravity_tpu",
        test_module="tt_top_cocotb",
        test_dir=HERE,
        build_dir=build_dir,
    )
