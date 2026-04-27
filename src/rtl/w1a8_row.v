/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * w1a8_row - horizontal chain of W1A8 systolic processing elements.
 *
 * Partial sums and valid move left to right, one PE per cycle. Activations
 * enter per column and are forwarded through each PE for the next array row.
 *
 * To compute one COLS-wide dot product, the controller must skew activation
 * lanes by column:
 *
 *   cycle 0: col0 sees x0, valid_in=1, psum_in=seed
 *   cycle 1: col1 sees x1
 *   cycle 2: col2 sees x2
 *   ...
 *
 * The final psum appears at psum_out when valid_out is high.
 */

`default_nettype none

module w1a8_row #(
    parameter ACT_WIDTH  = 8,
    parameter PSUM_WIDTH = 16,
    parameter COLS       = 4
) (
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire                              clear,
    input  wire                              weight_load,
    input  wire                              weight_in,

    input  wire signed [COLS*ACT_WIDTH-1:0] act_in,
    output wire signed [COLS*ACT_WIDTH-1:0] act_out,

    input  wire signed [PSUM_WIDTH-1:0]      psum_in,
    input  wire                              valid_in,
    output wire signed [PSUM_WIDTH-1:0]     psum_out,
    output wire                              valid_out
);

    wire [COLS:0] weight_chain;
    wire [COLS:0] valid_chain;
    wire signed [PSUM_WIDTH-1:0] psum_chain [0:COLS];

    assign weight_chain[0] = weight_in;
    assign valid_chain[0] = valid_in;
    assign valid_out      = valid_chain[COLS];

    assign psum_chain[0] = psum_in;
    assign psum_out      = psum_chain[COLS];

    genvar col;
    generate
        for (col = 0; col < COLS; col = col + 1) begin : gen_col
            w1a8_pe #(
                .ACT_WIDTH (ACT_WIDTH),
                .PSUM_WIDTH(PSUM_WIDTH)
            ) u_pe (
                .clk        (clk),
                .rst_n      (rst_n),
                .clear      (clear),
                .weight_load(weight_load),
                .weight_in  (weight_chain[col]),
                .weight_out (weight_chain[col + 1]),
                .act_in     (act_in[col*ACT_WIDTH +: ACT_WIDTH]),
                .psum_in    (psum_chain[col]),
                .valid_in   (valid_chain[col]),
                .act_out    (act_out[col*ACT_WIDTH +: ACT_WIDTH]),
                .psum_out   (psum_chain[col + 1]),
                .valid_out  (valid_chain[col + 1])
            );
        end
    endgenerate

endmodule
