SIM ?= verilator
TOPLEVEL_LANG ?= verilog
TOPLEVEL = q1a8_reducer
COCOTB_TEST_MODULES = test

VERILOG_SOURCES = \
    $(PWD)/fp16_to_fp32.v \
    $(PWD)/int_to_fp32.v \
    $(PWD)/fp32_mul.v \
    $(PWD)/q1a8_reducer.v

ifeq ($(SIM),verilator)
EXTRA_ARGS += --trace --trace-structs
endif

include $(shell cocotb-config --makefiles)/Makefile.sim
