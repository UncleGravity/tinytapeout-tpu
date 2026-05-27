// Q1A8 reducer - one Q8 sub-block per cycle.
//
// For each cycle valid_in is high, consumes
//   - 32 weight bits (one bit per output activation)
//   - 32 signed int8 activations
//   - fp16 weight_scale  (constant across the 4 Q8 sub-blocks of one Q1 block)
//   - fp16 act_scale     (varies per Q8 sub-block)
//
// and produces (two cycles later)
//
//   contribution = (fp32) weight_scale * act_scale * Sigma_i (b_i ? +a_i : -a_i)
//
// which is exactly one term in the Q1A8 matmul inner sum from
// tests/golden/kernels.py:matmul_q1a8. An outer module accumulates these
// contributions into one output lane over K/32 cycles.
//
// All math after the integer reduce is fp32 (round-toward-zero - see
// fp32_mul.v). The two fp32 multiplies are separated by a register stage
// so the rowblock version can close timing when several lanes are replicated.
//
// Latency:    2 cycles
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

    fp16_to_fp32 u_ws  (.in(weight_scale), .out(weight_scale_f32));
    fp16_to_fp32 u_as  (.in(act_scale),    .out(act_scale_f32));

    // -- Pipeline stage 1 --------------------------------------------------
    reg         valid_s1;
    reg [31:0] weight_scale_f32_s1;
    reg [31:0] act_scale_f32_s1;
    reg signed [13:0] sub_sum_s1;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s1            <= 1'b0;
            weight_scale_f32_s1 <= 32'd0;
            act_scale_f32_s1    <= 32'd0;
            sub_sum_s1          <= 14'sd0;
        end else begin
            valid_s1            <= valid_in;
            weight_scale_f32_s1 <= weight_scale_f32;
            act_scale_f32_s1    <= act_scale_f32;
            sub_sum_s1          <= sub_sum;
        end
    end

    // -- Pipeline stage 2 --------------------------------------------------
    // Fold the two scales first so the mantissa product is the small one
    // (scale * scale). Then multiply by sub_sum_f32 - this is the larger-
    // magnitude factor.
    wire [31:0] combined_f32;
    wire [31:0] sub_sum_f32;

    fp32_mul u_combine (.a(weight_scale_f32_s1), .b(act_scale_f32_s1),
                        .out(combined_f32));
    int_to_fp32 #(.WIDTH(14)) u_int (.in(sub_sum_s1), .out(sub_sum_f32));

    reg         valid_s2;
    reg [31:0] combined_f32_s2;
    reg [31:0] sub_sum_f32_s2;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_s2        <= 1'b0;
            combined_f32_s2 <= 32'd0;
            sub_sum_f32_s2  <= 32'd0;
        end else begin
            valid_s2        <= valid_s1;
            combined_f32_s2 <= combined_f32;
            sub_sum_f32_s2  <= sub_sum_f32;
        end
    end

    // -- Pipeline stage 3 --------------------------------------------------
    wire [31:0] contribution_comb;

    fp32_mul u_contrib (.a(combined_f32_s2),  .b(sub_sum_f32_s2),
                        .out(contribution_comb));

    // -- Output register ---------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out    <= 1'b0;
            contribution <= 32'd0;
        end else begin
            valid_out    <= valid_s2;
            contribution <= contribution_comb;
        end
    end

endmodule
