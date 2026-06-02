// Truncating (round-toward-zero) IEEE 754 fp32 multiplier.
//
// Simplifications relative to a fully IEEE-754 compliant multiplier:
//   - Round-toward-zero (max 1 ULP off vs round-to-nearest-even).
//   - Subnormal inputs and outputs flush to signed zero.
//   - Overflow saturates to the max normal value (NOT IEEE Inf).
//   - NaN inputs not detected; they produce garbage.
//
// These are all fine for W1A8 where scales and sub_sums are bounded and
// finite. The eventual release bitstream should swap this for the Xilinx
// Floating Point Operator IP, which is fully compliant and uses DSP slices
// natively. This module exists primarily so the simulation testbench can
// match the hardware bit-for-bit without depending on Vivado IP at sim time.

`default_nettype none

module fp32_mul (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] out
);
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] ma = a[22:0];
    wire        sb = b[31];
    wire [7:0]  eb = b[30:23];
    wire [22:0] mb = b[22:0];

    wire is_zero = (ea == 8'd0) || (eb == 8'd0);

    // Prepend the implicit "1." hidden bit and multiply 24x24 -> 48 bits.
    // The product is in [1.0, 4.0); bit 47 set means it's in [2.0, 4.0) and
    // we need to renormalize by shifting right one place.
    // use_dsp forces the 24x24 onto DSP48 slices — at 150 MHz a LUT-fabric
    // multiply on this path will not close timing. (Verilator ignores the
    // attribute, so the sim stays bit-exact.)
    (* use_dsp = "yes" *) wire [47:0] prod = {1'b1, ma} * {1'b1, mb};
    wire        renorm = prod[47];
    wire [22:0] mr     = renorm ? prod[46:24] : prod[45:23];

    // Combined exponent, signed so we can detect under/overflow.
    wire signed [9:0] er_pre = $signed({2'b00, ea})
                             + $signed({2'b00, eb})
                             - 10'sd127;
    wire signed [9:0] er     = er_pre + (renorm ? 10'sd1 : 10'sd0);

    wire sr        = sa ^ sb;
    wire underflow = (er <= 10'sd0);
    wire overflow  = (er >= 10'sd255);

    assign out =
        is_zero    ? 32'd0 :
        underflow  ? {sr, 31'd0} :
        overflow   ? {sr, 8'hFE, 23'h7FFFFF} :   // max normal, not Inf
                     {sr, er[7:0], mr};
endmodule
