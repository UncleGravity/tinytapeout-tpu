set script_dir [file normalize [file dirname [info script]]]
set out_dir [file normalize "$script_dir/out"]
set proj_dir [file normalize "$script_dir/build"]
set proj_name dma_loopback
set bd_name dma_loopback_bd

file delete -force $out_dir
file mkdir $out_dir

create_project $proj_name $proj_dir -part xc7z020clg400-1 -force
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
set_property target_language Verilog [current_project]

add_files -norecurse "$script_dir/src/axis_loopback.v"
update_compile_order -fileset sources_1

create_bd_design $bd_name

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

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_0

create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inv_0
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] [get_bd_cells reset_inv_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* const_1
set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_1]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_s_axis_s2mm_tdata_width {64} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
] [get_bd_cells axi_dma_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_lite_interconnect
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells axi_lite_interconnect]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_mem_interconnect
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {2}] [get_bd_cells axi_mem_interconnect]

create_bd_cell -type module -reference axis_loopback axis_loopback_0

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins rst_0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins ps7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins ps7_0/S_AXI_HP0_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins reset_inv_0/Op1]
connect_bd_net [get_bd_pins reset_inv_0/Res] [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins const_1/dout] [get_bd_pins rst_0/dcm_locked]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axis_loopback_0/aclk]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axis_loopback_0/aresetn]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_lite_interconnect/ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_lite_interconnect/S00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_lite_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins axi_lite_interconnect/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_lite_interconnect/S00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_lite_interconnect/M00_ARESETN]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect/ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect/S00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect/S01_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins axi_mem_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins axi_mem_interconnect/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_mem_interconnect/S00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_mem_interconnect/S01_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_mem_interconnect/M00_ARESETN]

connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] [get_bd_intf_pins axi_lite_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite_interconnect/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins axi_mem_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins axi_mem_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_interconnect/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins axis_loopback_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_loopback_0/M_AXIS] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

set wrapper [make_wrapper -files [get_files "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"] -top]
add_files -norecurse $wrapper
update_compile_order -fileset sources_1

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

file copy -force "$proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit" "$out_dir/dma_loopback.bit"
file copy -force "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh" "$out_dir/dma_loopback.hwh"

puts "WROTE $out_dir/dma_loopback.bit"
puts "WROTE $out_dir/dma_loopback.hwh"
