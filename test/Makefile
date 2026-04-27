# Makefile
# See https://docs.cocotb.org/en/stable/quickstart.html for more info

# defaults
SIM ?= icarus
FST ?= -fst # Use more efficient FST format
TOPLEVEL_LANG ?= verilog
SRC_DIR = $(PWD)/../src
INFO_YAML = $(PWD)/../info.yaml

# Derive top module and source files from info.yaml
TOP_MODULE := $(shell grep 'top_module:' $(INFO_YAML) | sed 's/.*: *"\(.*\)"/\1/')
PROJECT_SOURCES := $(shell sed -n '/^  source_files:/,/^[^ ]/{ /^ *- /{ s/.*- *"\(.*\)"/\1/; p; }; }' $(INFO_YAML))

# Derive PDK_ROOT from flake layout if not already set
# ifndef PDK_ROOT
#   $(error PDK_ROOT is not set)
# endif

# Gate-level netlist: use local copy if present, otherwise pull from hardening run
GATE_LEVEL_NETLIST ?= $(PWD)/gate_level_netlist.v
ifeq ($(wildcard $(GATE_LEVEL_NETLIST)),)
  GATE_LEVEL_NETLIST = $(PWD)/../runs/wokwi/final/pnl/$(TOP_MODULE).pnl.v
endif

ifneq ($(GATES),yes)

# RTL simulation:
SIM_BUILD				= ./sim_build/rtl
VERILOG_SOURCES += $(addprefix $(SRC_DIR)/,$(PROJECT_SOURCES))

else

# Gate level simulation:
SIM_BUILD				= ./sim_build/gl
COMPILE_ARGS    += -DGL_TEST
COMPILE_ARGS    += -DFUNCTIONAL
COMPILE_ARGS    += -DUSE_POWER_PINS
COMPILE_ARGS    += -DSIM
COMPILE_ARGS    += -DUNIT_DELAY=\#1
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/primitives.v
VERILOG_SOURCES += $(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/verilog/sky130_fd_sc_hd.v

# this gets copied in by the GDS action workflow
VERILOG_SOURCES += $(GATE_LEVEL_NETLIST)

endif

# Waveform dump location
COMPILE_ARGS    += -DDUMPFILE=\"$(SIM_BUILD)/tb.fst\"

# Current RTL uses SystemVerilog constructs such as always_ff/always_comb.
COMPILE_ARGS    += -g2012

# Allow sharing configuration between design and testbench via `include`:
COMPILE_ARGS 		+= -I$(SRC_DIR)

# Include the testbench sources:
VERILOG_SOURCES += $(PWD)/tb.v
TOPLEVEL = tb

# List test modules to run, separated by commas and without the .py suffix:
COCOTB_TEST_MODULES = integration

# include cocotb's make rules to take care of the simulator setup
include $(shell cocotb-config --makefiles)/Makefile.sim
