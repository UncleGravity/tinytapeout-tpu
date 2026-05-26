# vivado -mode batch -source tcl/build.tcl
#
# matmul_q1a8 bitstream: ps7 + axi_dma (MM2S) + q1a8_kernel_top, with the
# host driving control via the PS GP0 (AXI-Lite) and data via AXI DMA over
# HP0 (AXI HP).
#
# Topology:
#   ps7/M_AXI_GP0  -> axi_lite_interconnect -> { axi_dma.S_AXI_LITE,
#                                                q1a8_kernel_top.S_AXI }
#   axi_dma.M_AXI_MM2S -> axi_mem_interconnect -> ps7/S_AXI_HP0
#   axi_dma.M_AXIS_MM2S -> q1a8_kernel_top.S_AXIS
#
# build.sh stages this folder AND the shared `rtl/` tree at the Vivado
# work root, so paths like ../rtl/q1a8/*.v resolve on the VM the same way
# they do locally.

set script_dir [file normalize [file dirname [info script]]]
set proj_root  [file normalize [file join $script_dir ..]]
set rtl_dir    [file join $proj_root rtl q1a8]
set out_dir    [file join $proj_root out]
set proj_dir   [file join $out_dir vivado]
set proj_name  matmul_q1a8
set bd_name    matmul_q1a8_bd

file delete -force $out_dir
file mkdir $out_dir

create_project $proj_name $proj_dir -part xc7z020clg400-1 -force
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
set_property target_language Verilog [current_project]

# -- Add all q1a8 RTL files --------------------------------------------------
foreach v {
    fp16_to_fp32.v
    int_to_fp32.v
    fp32_mul.v
    fp32_add.v
    q1a8_reducer.v
    q1a8_cell.v
    q1a8_streamer.v
    axis_to_subblock.v
    q1a8_kernel.v
    q1a8_kernel_top.v
} {
    add_files -norecurse [file join $rtl_dir $v]
}
update_compile_order -fileset sources_1

# -- Block design -----------------------------------------------------------

create_bd_design $bd_name

# Processing system (PS7).
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* ps7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7_0]

# Reset generator.
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_0
create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inv_0
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] [get_bd_cells reset_inv_0]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* const_1
set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_1]

# AXI DMA - MM2S only (read from DDR, push as AXIS to the kernel).
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
] [get_bd_cells axi_dma_0]

# AXI-Lite interconnect: 1 SI (GP0) -> 2 MI (DMA control + kernel control).
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_lite_interconnect
set_property -dict [list CONFIG.NUM_MI {2} CONFIG.NUM_SI {1}] [get_bd_cells axi_lite_interconnect]

# Memory interconnect: 1 SI (DMA MM2S) -> 1 MI (HP0).
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_mem_interconnect
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells axi_mem_interconnect]

# Custom kernel top (Vivado infers the AXI-Lite slave + AXIS slave from
# the X_INTERFACE_INFO attributes in the RTL).
create_bd_cell -type module -reference q1a8_kernel_top q1a8_kernel_top_0

# --- DIAGNOSTIC: what interfaces did Vivado infer on the custom cell? ----
puts "==============================================================="
puts "DIAG: q1a8_kernel_top_0 INTERFACE PINS (post create_bd_cell):"
foreach pin [get_bd_intf_pins -of_objects [get_bd_cells q1a8_kernel_top_0]] {
    puts "  intf: $pin"
}
puts "DIAG: q1a8_kernel_top_0 SCALAR PINS:"
foreach pin [get_bd_pins -of_objects [get_bd_cells q1a8_kernel_top_0]] {
    puts "  pin:  $pin"
}
puts "==============================================================="

# -- Clocks + resets --------------------------------------------------------

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins rst_0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins ps7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]     [get_bd_pins ps7_0/S_AXI_HP0_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins reset_inv_0/Op1]
connect_bd_net [get_bd_pins reset_inv_0/Res]     [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins const_1/dout]        [get_bd_pins rst_0/dcm_locked]

# DMA
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]

# Kernel top
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins q1a8_kernel_top_0/s_axi_aclk]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins q1a8_kernel_top_0/s_axi_aresetn]

# AXI-Lite interconnect (1 SI + 2 MI)
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_lite_interconnect/ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_lite_interconnect/S00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_lite_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_lite_interconnect/M01_ACLK]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins axi_lite_interconnect/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_lite_interconnect/S00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_lite_interconnect/M00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_lite_interconnect/M01_ARESETN]

# Memory interconnect (1 SI + 1 MI)
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_mem_interconnect/ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_mem_interconnect/S00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]            [get_bd_pins axi_mem_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins axi_mem_interconnect/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_mem_interconnect/S00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_mem_interconnect/M00_ARESETN]

# -- AXI plumbing -----------------------------------------------------------

# Control: GP0 -> AXI-Lite interconnect -> { DMA control, kernel control }
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0]               [get_bd_intf_pins axi_lite_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite_interconnect/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_lite_interconnect/M01_AXI] [get_bd_intf_pins q1a8_kernel_top_0/S_AXI]

# Data: DMA MM2S read -> mem interconnect -> HP0
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S]          [get_bd_intf_pins axi_mem_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_interconnect/M00_AXI]  [get_bd_intf_pins ps7_0/S_AXI_HP0]

# Stream: DMA MM2S stream -> kernel data input
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S]         [get_bd_intf_pins q1a8_kernel_top_0/S_AXIS]

assign_bd_address
validate_bd_design
save_bd_design

# --- DIAGNOSTIC: what's actually exposed at the BD boundary? -------------
puts "==============================================================="
puts "DIAG: BD external ports (these will become wrapper top-level pins):"
foreach port [get_bd_ports] {
    puts "  port: $port"
}
puts "DIAG: BD external interface ports:"
foreach iport [get_bd_intf_ports] {
    puts "  intf: $iport"
}
puts "==============================================================="

# -- Wrap and build ---------------------------------------------------------

set wrapper [make_wrapper -files [get_files "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"] -top]
add_files -norecurse $wrapper
update_compile_order -fileset sources_1
set_property top ${bd_name}_wrapper [current_fileset]

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "synth_1 did not complete"
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "impl_1 did not complete"
}

file copy -force "$proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit" "$out_dir/matmul_q1a8.bit"
file copy -force "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh" "$out_dir/matmul_q1a8.hwh"

puts "==> Built: $out_dir/matmul_q1a8.{bit,hwh}"
