// q1a8_kernel - everything-but-AXI-Lite top.
//
// Composes the AXIS packer with the streamer, presenting the two
// interfaces a Vivado block design will eventually wire up:
//
//   - Kernel control     (becomes AXI-Lite registers in the bitstream)
//   - 64-bit AXIS input  (connects to the M_AXIS_MM2S port of an AXI DMA)
//
// No AXI-Lite slave here - that gets added when we move to Vivado, using
// the same `axi_lite_regs` pattern as the `axi_lite_probe` design. The
// register map will be a thin shim over these control ports.

`default_nettype none

module q1a8_kernel (
    input  wire         clk,
    input  wire         rst_n,

    // Kernel control (-> AXI-Lite later).
    input  wire         start_kernel,
    input  wire [15:0]  num_subblocks,
    output wire         kernel_done,
    output wire         busy,
    output wire [31:0]  result,

    // 64-bit AXIS data input (-> AXI DMA M_AXIS_MM2S).
    input  wire [63:0]  s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready
);
    // -- Internal sub-block bus (packer -> streamer) ---------------------
    wire         sb_valid;
    wire         sb_ready;
    wire [31:0]  sb_weight_bits;
    wire [255:0] sb_acts_packed;
    wire [15:0]  sb_weight_scale;
    wire [15:0]  sb_act_scale;

    axis_to_subblock u_packer (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_valid(sb_valid),
        .m_ready(sb_ready),
        .m_weight_bits(sb_weight_bits),
        .m_acts_packed(sb_acts_packed),
        .m_weight_scale(sb_weight_scale),
        .m_act_scale(sb_act_scale)
    );

    q1a8_streamer u_streamer (
        .clk(clk), .rst_n(rst_n),
        .start_kernel(start_kernel),
        .num_subblocks(num_subblocks),
        .kernel_done(kernel_done),
        .busy(busy),
        .result(result),
        .s_valid(sb_valid),
        .s_ready(sb_ready),
        .s_weight_bits(sb_weight_bits),
        .s_acts_packed(sb_acts_packed),
        .s_weight_scale(sb_weight_scale),
        .s_act_scale(sb_act_scale)
    );
endmodule
