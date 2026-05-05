Verilog project that can be hardened into:
- GDS: via tinytapeout helpers or directly with librelane
- TinyTapeout FPGA (iCE40UP5K) dev board

All commands must be run inside of nix dev shell
- eg. `nix develop -c <command>`

Most tinytapeout actions will happen through the tt_tool.py script
`tt/tt_tool.py --help`

Hardening for TT:
`tt/tt_tool.py --create-user-config`
`tt/tt_tool.py --harden --no-docker`

Hardening for FPGA:
`tt/tt_fpga.py harden`
`tt/tt_fpga.py configure --port /dev/<tty> --upload --clockrate 12000000`

Testing:
RTL: `cd test && make -B`
GATE: `cd test && make -B GATES=yes`

More Info:
https://tinytapeout.com/guides/local-hardening/
https://tinytapeout.com/guides/fpga-breakout/
https://librelane.readthedocs.io/en/stable/

# Verilog
- Use always_ff @(posedge clk) for sequential logic. synthesis will complain if you accidentally create combinational paths inside it.
- Use always_comb instead of always @(*) for combinational logic. Synthesis will complain about inferred latches.
- Keep systemverilog features to a minimum to avoid yosys issues.

# C++
Use plain structs for data
Use free functions for pure-ish transformations/checks
Use OOP only when there is state, ownership, or a replaceable backend boundary

# Tests
Integration tests live in test/
Unit tests live in test/unit/
