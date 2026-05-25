SIM ?= verilator
TOPLEVEL_LANG ?= verilog
TOPLEVEL = q1a8_streamer
COCOTB_TEST_MODULES = test

REDUCER_DIR = $(PWD)/../q1a8_reducer
CELL_DIR    = $(PWD)/../q1a8_cell

VERILOG_SOURCES = \
    $(REDUCER_DIR)/fp16_to_fp32.v \
    $(REDUCER_DIR)/int_to_fp32.v \
    $(REDUCER_DIR)/fp32_mul.v \
    $(REDUCER_DIR)/q1a8_reducer.v \
    $(CELL_DIR)/fp32_add.v \
    $(CELL_DIR)/q1a8_cell.v \
    $(PWD)/q1a8_streamer.v

ifeq ($(SIM),verilator)
EXTRA_ARGS += --trace --trace-structs
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
