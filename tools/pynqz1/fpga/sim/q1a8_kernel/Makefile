SIM ?= verilator
TOPLEVEL_LANG ?= verilog
TOPLEVEL = q1a8_kernel
COCOTB_TEST_MODULES = test

RTL = $(PWD)/../../rtl/q1a8

VERILOG_SOURCES = \
    $(RTL)/fp16_to_fp32.v \
    $(RTL)/int_to_fp32.v \
    $(RTL)/fp32_mul.v \
    $(RTL)/q1a8_reducer.v \
    $(RTL)/fp32_add.v \
    $(RTL)/q1a8_cell.v \
    $(RTL)/q1a8_streamer.v \
    $(RTL)/axis_to_subblock.v \
    $(RTL)/q1a8_kernel.v

ifeq ($(SIM),verilator)
EXTRA_ARGS += --trace --trace-structs
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
