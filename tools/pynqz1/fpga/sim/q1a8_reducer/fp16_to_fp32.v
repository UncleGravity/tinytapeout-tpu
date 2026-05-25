// fp16 -> fp32 conversion.
//
// Designed for quantization scales (positive, normal range). Subnormals
// flush to signed zero; Inf/NaN aren't expected and produce garbage. Both
// behaviors are fine for W1A8 because Q1_0 / Q8_0 scales come from amax/127
// of finite inputs, so they're always finite and either zero or normal.

`default_nettype none

module fp16_to_fp32 (
    input  wire [15:0] in,
    output wire [31:0] out
);
    wire        sign    = in[15];
    wire [4:0]  exp_in  = in[14:10];
    wire [9:0]  mant_in = in[9:0];

    // Subnormal (exp_in==0) -> flush to signed zero.
    wire is_zero_sub = (exp_in == 5'd0);

    // Normal: rebias exponent (15 -> 127, so +112), zero-pad mantissa.
    // Zero-extend exp_in to 8 bits explicitly so verilator's WIDTHEXPAND
    // lint is happy.
    wire [7:0]  exp_out  = {3'd0, exp_in} + 8'd112;
    wire [22:0] mant_out = {mant_in, 13'd0};

    assign out = is_zero_sub ? {sign, 31'd0}
                             : {sign, exp_out, mant_out};
endmodule
