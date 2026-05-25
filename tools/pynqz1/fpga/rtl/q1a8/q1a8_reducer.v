// Q1A8 reducer - one Q8 sub-block per cycle.
//
// For each cycle valid_in is high, consumes
//   - 32 weight bits (one bit per output activation)
//   - 32 signed int8 activations
//   - fp16 weight_scale  (constant across the 4 Q8 sub-blocks of one Q1 block)
//   - fp16 act_scale     (varies per Q8 sub-block)
//
// and produces (one cycle later)
//
//   contribution = (fp32) weight_scale * act_scale * Sigma_i (b_i ? +a_i : -a_i)
//
// which is exactly one term in the Q1A8 matmul inner sum from
// tests/golden/kernels.py:matmul_q1a8. An outer module accumulates these
// contributions into one output cell over K/32 cycles.
//
// All math after the integer reduce is fp32 (round-toward-zero - see
// fp32_mul.v). The combinational depth is dominated by the 32-wide signed
// add tree plus two fp32 multiplies; tight at 100 MHz but fine for sim.
// When timing closes badly, the natural place to add pipeline stages is
// before/after `combined_f32` and before `contribution`.
//
// Latency:    1 cycle (inputs latched on rising edge, output valid next edge)
// Throughput: 1 sub-block / cycle

`default_nettype none

module q1a8_reducer (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         valid_in,

    input  wire [31:0]  weight_bits,   // bit i pairs with acts[i]
    input  wire [255:0] acts_packed,   // 32 x int8, LSB-first
    input  wire [15:0]  weight_scale,  // fp16
    input  wire [15:0]  act_scale,     // fp16

    output reg          valid_out,
    output reg  [31:0]  contribution   // fp32
);

    // -- Integer reduction -------------------------------------------------
    // sub_sum range: 32 * 128 = 4096 magnitude (when all weights=0 and all
    // acts=-128, the conditional negation produces +128 per element).
    // 14-bit signed covers [-8192, +8191] - one bit of headroom.
    //
    // Each int8 act is sign-extended to int14 *before* the conditional
    // negate. That way both branches of the conditional are 14-bit signed
    // and verilator's WIDTHEXPAND lint stays quiet.
    function automatic signed [13:0] sext_act;
        input [7:0] a;
        sext_act = $signed({{6{a[7]}}, a});
    endfunction

    integer i;
    reg signed [13:0] sub_sum;
    always @(*) begin
        sub_sum = 14'sd0;
        for (i = 0; i < 32; i = i + 1) begin
            sub_sum = sub_sum + (weight_bits[i] ?
                 sext_act(acts_packed[i*8 +: 8]) :
                -sext_act(acts_packed[i*8 +: 8]));
        end
    end

    // -- Format conversion -------------------------------------------------
    wire [31:0] weight_scale_f32;
    wire [31:0] act_scale_f32;
    wire [31:0] sub_sum_f32;

    fp16_to_fp32 u_ws  (.in(weight_scale), .out(weight_scale_f32));
    fp16_to_fp32 u_as  (.in(act_scale),    .out(act_scale_f32));
    int_to_fp32 #(.WIDTH(14)) u_int (.in(sub_sum), .out(sub_sum_f32));

    // -- fp32 multiplies ---------------------------------------------------
    // Fold the two scales first so the mantissa product is the small one
    // (scale * scale). Then multiply by sub_sum_f32 - this is the larger-
    // magnitude factor and lives on the critical path.
    wire [31:0] combined_f32;
    wire [31:0] contribution_comb;

    fp32_mul u_combine (.a(weight_scale_f32), .b(act_scale_f32),
                        .out(combined_f32));
    fp32_mul u_contrib (.a(combined_f32),     .b(sub_sum_f32),
                        .out(contribution_comb));

    // -- Output register ---------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out    <= 1'b0;
            contribution <= 32'd0;
        end else begin
            valid_out    <= valid_in;
            contribution <= contribution_comb;
        end
    end

endmodule
