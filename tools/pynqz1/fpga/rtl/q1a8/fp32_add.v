// fp32_add - truncating IEEE 754 fp32 addition.
//
// Simplifications relative to fully IEEE-754 compliant addition:
//   - Round toward zero (no nearest-even; at most 1 ULP off).
//   - Subnormal inputs and outputs flush to signed zero.
//   - Overflow saturates to the max normal value (not Inf).
//   - NaN inputs aren't detected and produce garbage.
//
// All fine for the W1A8 accumulator: contributions are bounded products
// of small scales and bounded sub_sums, the running sum stays finite, and
// there's no need to model Inf/NaN.
//
// Pairs with the Python `fp32_add_trunc` in the q1a8 kernel tests - both
// follow exactly the same code path so the comparison is bit-exact.

`default_nettype none

module fp32_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] out
);
    // -- Decode -----------------------------------------------------------
    wire        sa = a[31];
    wire [7:0]  ea = a[30:23];
    wire [22:0] ma = a[22:0];
    wire        sb = b[31];
    wire [7:0]  eb = b[30:23];
    wire [22:0] mb = b[22:0];

    wire a_zero = (ea == 8'd0);
    wire b_zero = (eb == 8'd0);

    // -- Pick "big" = larger-exp side; "small" = the other --------------
    wire        a_ge_b     = (ea >= eb);
    wire [7:0]  exp_big    = a_ge_b ? ea : eb;
    wire [7:0]  exp_small  = a_ge_b ? eb : ea;
    wire [23:0] mant_big   = a_ge_b ? {1'b1, ma} : {1'b1, mb};
    wire [23:0] mant_small = a_ge_b ? {1'b1, mb} : {1'b1, ma};
    wire        sign_big   = a_ge_b ? sa : sb;
    wire        sign_small = a_ge_b ? sb : sa;

    wire [7:0]  exp_diff   = exp_big - exp_small;

    // -- Align small to big by right-shifting ----------------------------
    // If diff > 24, small contribution rounds below big's LSB; flush to 0.
    wire [23:0] mant_small_aligned = (exp_diff > 8'd24)
                                       ? 24'd0
                                       : (mant_small >> exp_diff[4:0]);

    // -- Resolve same-exp magnitude tie ----------------------------------
    // After alignment, big's magnitude >= small's UNLESS exp_diff==0 and
    // small's raw mantissa was bigger. In that one corner case, swap so the
    // "big" slot really has the larger magnitude (so the subtract below
    // never goes negative).
    wire same_exp          = (exp_diff == 8'd0);
    wire small_mant_bigger = same_exp && (mant_small_aligned > mant_big);

    wire [23:0] m1, m2;
    wire        result_sign;
    assign {m1, m2, result_sign} = small_mant_bigger
        ? {mant_small_aligned, mant_big,           sign_small}
        : {mant_big,           mant_small_aligned, sign_big};

    // -- Add or subtract magnitudes (one extra bit for carry) ------------
    wire same_sign = (sign_big == sign_small);
    wire [24:0] mant_sum = same_sign
        ? ({1'b0, m1} + {1'b0, m2})
        : ({1'b0, m1} - {1'b0, m2});

    wire sum_zero = (mant_sum == 25'd0);

    // -- Find leading 1 in mant_sum (0..24) ------------------------------
    // Same priority-encoder pattern as int_to_fp32 - synthesizes fine.
    reg [4:0] lead_pos;
    integer ii;
    always @(*) begin
        lead_pos = 5'd0;
        for (ii = 0; ii <= 24; ii = ii + 1) begin
            if (mant_sum[ii]) lead_pos = ii[4:0];
        end
    end

    // -- Normalize so leading 1 lands at bit 23 --------------------------
    wire       shift_right  = (lead_pos > 5'd23);
    wire [4:0] right_amount = shift_right ? (lead_pos - 5'd23) : 5'd0;
    wire [4:0] left_amount  = (lead_pos < 5'd23) ? (5'd23 - lead_pos) : 5'd0;

    wire [24:0] mant_normalized = shift_right
        ? (mant_sum >> right_amount)
        : (mant_sum << left_amount);

    // -- Compute new exponent (signed for under/overflow detection) ------
    wire signed [9:0] exp_signed = shift_right
        ? ($signed({2'b00, exp_big}) + $signed({5'd0, right_amount}))
        : ($signed({2'b00, exp_big}) - $signed({5'd0, left_amount}));

    wire underflow = sum_zero || (exp_signed <= 10'sd0);
    wire overflow  = (exp_signed >= 10'sd255);

    // -- Assemble result (with zero / sat shortcuts) ---------------------
    wire [22:0] result_mantissa = mant_normalized[22:0];
    wire [7:0]  result_exponent = exp_signed[7:0];

    assign out = a_zero    ? b
               : b_zero    ? a
               : underflow ? {result_sign, 31'd0}
               : overflow  ? {result_sign, 8'hFE, 23'h7FFFFF}
                           : {result_sign, result_exponent, result_mantissa};
endmodule
