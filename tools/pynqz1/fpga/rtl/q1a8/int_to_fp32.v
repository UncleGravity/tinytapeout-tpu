// Signed integer (parameterized width) -> IEEE 754 fp32.
//
// Truncating (round-toward-zero). For W1A8 the input is the Q8 sub-block
// sub_sum bounded in [-4096, +4096], so WIDTH=14 covers it with one bit of
// headroom and the magnitude always fits in the 23-bit fp32 mantissa
// without any rounding at all.
//
// Caveat: the most-negative WIDTH-bit value (-2^(WIDTH-1)) negates to itself
// in two's complement. We never reach that point for W1A8 (|sub_sum| <= 4096
// < 8192), but if you instantiate this with values that can reach the
// negative-extreme, widen WIDTH by one or guard the input.

`default_nettype none

module int_to_fp32 #(
    parameter integer WIDTH = 14
) (
    input  wire signed [WIDTH-1:0] in,
    output wire        [31:0]      out
);
    localparam integer MAG_W = WIDTH - 1;

    wire                    sign    = in[WIDTH-1];
    wire signed [WIDTH-1:0] neg_in  = -in;
    wire [MAG_W-1:0]        mag     = sign ? neg_in[MAG_W-1:0]
                                           : in[MAG_W-1:0];

    // Priority encoder: find the position of the leading 1 in `mag`.
    // Synthesizes to a chain of muxes; combinational.
    reg [4:0] msb_pos;
    integer i;
    always @(*) begin
        msb_pos = 5'd0;
        for (i = 0; i < MAG_W; i = i + 1) begin
            if (mag[i]) msb_pos = i[4:0];
        end
    end

    // Normalize: shift the leading 1 up to bit MAG_W-1, then drop it
    // (it's the IEEE 754 implicit hidden bit). Right-pad to 23 mantissa bits.
    // The shift count is computed in 32 bits (matching the implicit width
    // of `MAG_W - 1`) with msb_pos zero-extended; the shift operator accepts
    // any RHS width without WIDTHEXPAND complaints.
    wire [MAG_W-1:0] mag_norm = mag << ((MAG_W - 1) - {27'd0, msb_pos});
    wire [22:0]      mantissa = {mag_norm[MAG_W-2:0],
                                  {(23 - (MAG_W-1)){1'b0}}};

    wire [7:0] exponent = 8'd127 + {3'd0, msb_pos};
    wire       is_zero  = (mag == {MAG_W{1'b0}});

    assign out = is_zero ? 32'd0 : {sign, exponent, mantissa};
endmodule
