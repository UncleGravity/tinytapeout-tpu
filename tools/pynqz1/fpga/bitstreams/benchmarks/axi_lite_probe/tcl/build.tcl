# vivado -mode batch -source tcl/build.tcl
#
# Minimal block design: PS GP0 -> AXI-Lite interconnect -> axi_lite_regs.
# No DMA, no HP ports. The only PL logic is our hand-written register file.
#
# Produces out/axi_lite_probe.{bit,hwh}.

set script_dir [file normalize [file dirname [info script]]]
set proj_root  [file normalize [file join $script_dir ..]]
set rtl_dir    [file join $proj_root rtl]
set out_dir    [file join $proj_root out]
set proj_dir   [file join $out_dir vivado]
set proj_name  axi_lite_probe
set bd_name    axi_lite_probe_bd

file delete -force $out_dir
file mkdir $out_dir

create_project $proj_name $proj_dir -part xc7z020clg400-1 -force
set_property board_part www.digilentinc.com:pynq-z1:part0:1.0 [current_project]
set_property target_language Verilog [current_project]

add_files -norecurse [file join $rtl_dir axi_lite_regs.v]
update_compile_order -fileset sources_1

# -- Block design --------------------------------------------------------

create_bd_design $bd_name

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:* ps7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells ps7_0]
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_0

create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* reset_inv_0
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] [get_bd_cells reset_inv_0]

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* const_1
set_property -dict [list CONFIG.CONST_VAL {1} CONFIG.CONST_WIDTH {1}] [get_bd_cells const_1]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* axi_lite_interconnect
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells axi_lite_interconnect]

# The custom slave. Vivado infers the AXI4-Lite interface from the
# X_INTERFACE_INFO attributes on the module ports.
create_bd_cell -type module -reference axi_lite_regs axi_lite_regs_0

# -- Clocks + resets -----------------------------------------------------

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins rst_0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins ps7_0/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N]      [get_bd_pins reset_inv_0/Op1]
connect_bd_net [get_bd_pins reset_inv_0/Res]          [get_bd_pins rst_0/ext_reset_in]
connect_bd_net [get_bd_pins const_1/dout]             [get_bd_pins rst_0/dcm_locked]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_lite_interconnect/ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_lite_interconnect/S00_ACLK]
connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_lite_interconnect/M00_ACLK]
connect_bd_net [get_bd_pins rst_0/interconnect_aresetn] [get_bd_pins axi_lite_interconnect/ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_lite_interconnect/S00_ARESETN]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn]   [get_bd_pins axi_lite_interconnect/M00_ARESETN]

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0]          [get_bd_pins axi_lite_regs_0/s_axi_aclk]
connect_bd_net [get_bd_pins rst_0/peripheral_aresetn] [get_bd_pins axi_lite_regs_0/s_axi_aresetn]

# -- AXI plumbing --------------------------------------------------------

connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0]               [get_bd_intf_pins axi_lite_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite_interconnect/M00_AXI] [get_bd_intf_pins axi_lite_regs_0/S_AXI]

assign_bd_address
validate_bd_design
save_bd_design

# -- Wrap and build ------------------------------------------------------

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

file copy -force "$proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit" "$out_dir/axi_lite_probe.bit"
file copy -force "$proj_dir/$proj_name.gen/sources_1/bd/$bd_name/hw_handoff/${bd_name}.hwh" "$out_dir/axi_lite_probe.hwh"

puts "==> Built: $out_dir/axi_lite_probe.bit and .hwh"
