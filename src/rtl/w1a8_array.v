/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * w1a8_array - ROWS x COLS weight-stationary systolic array.
 *
 * Composition contracts:
 *   - each row is a horizontal partial-sum pipeline;
 *   - activations flow vertically from row to row;
 *   - each row has an independent serial weight-load lane;
 *   - each row has an independent left-edge psum/valid input and right-edge
 *     psum/valid output.
 *
 * A controller must skew activation lanes by column and skew each row's
 * valid/psum seed by row so the wavefront reaches every PE at the right time.
 * Raw row outputs are staggered by row; a later controller/output buffer can
 * align them if a single "all rows valid" pulse is needed.
 */

`default_nettype none

module w1a8_array #(
    parameter ACT_WIDTH  = 8,
    parameter PSUM_WIDTH = 16,
    parameter ROWS       = 2,
    parameter COLS       = 4
) (
    input  wire                               clk,
    input  wire                               rst_n,

    input  wire                               clear,
    input  wire                               weight_load,
    input  wire [ROWS-1:0]                    weight_in,

    input  wire signed [COLS*ACT_WIDTH-1:0]   act_in,
    output wire signed [COLS*ACT_WIDTH-1:0]   act_out,

    input  wire signed [ROWS*PSUM_WIDTH-1:0]  psum_in,
    input  wire [ROWS-1:0]                    valid_in,
    output wire signed [ROWS*PSUM_WIDTH-1:0] psum_out,
    output wire [ROWS-1:0]                   valid_out
);

    wire signed [COLS*ACT_WIDTH-1:0] act_chain [0:ROWS];

    assign act_chain[0] = act_in;
    assign act_out      = act_chain[ROWS];

    genvar row;
    generate
        for (row = 0; row < ROWS; row = row + 1) begin : gen_row
            w1a8_row #(
                .ACT_WIDTH (ACT_WIDTH),
                .PSUM_WIDTH(PSUM_WIDTH),
                .COLS      (COLS)
            ) u_row (
                .clk        (clk),
                .rst_n      (rst_n),
                .clear      (clear),
                .weight_load(weight_load),
                .weight_in  (weight_in[row]),
                .act_in     (act_chain[row]),
                .act_out    (act_chain[row + 1]),
                .psum_in    (psum_in[row*PSUM_WIDTH +: PSUM_WIDTH]),
                .valid_in   (valid_in[row]),
                .psum_out   (psum_out[row*PSUM_WIDTH +: PSUM_WIDTH]),
                .valid_out  (valid_out[row])
            );
        end
    endgenerate

endmodule
