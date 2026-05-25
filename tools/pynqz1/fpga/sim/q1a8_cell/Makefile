SIM ?= verilator
TOPLEVEL_LANG ?= verilog
TOPLEVEL = q1a8_cell
COCOTB_TEST_MODULES = test

# Submodules under test live in the sibling q1a8_reducer/ sim folder so they
# can share the same source-of-truth. If/when we add an rtl/ tree, both
# tests will reference that instead.
REDUCER_DIR = $(PWD)/../q1a8_reducer

VERILOG_SOURCES = \
    $(REDUCER_DIR)/fp16_to_fp32.v \
    $(REDUCER_DIR)/int_to_fp32.v \
    $(REDUCER_DIR)/fp32_mul.v \
    $(REDUCER_DIR)/q1a8_reducer.v \
    $(PWD)/fp32_add.v \
    $(PWD)/q1a8_cell.v

ifeq ($(SIM),verilator)
EXTRA_ARGS += --trace --trace-structs
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
