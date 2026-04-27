/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * w1a8_pe - one processing element for a weight-stationary systolic array.
 *
 * The PE stores one 1-bit weight. During compute, it forwards the activation
 * and valid bit by one cycle while adding its signed contribution to the
 * incoming partial sum:
 *
 *   psum_out = psum_in + (weight ? +act_in : -act_in)
 *
 * weight_out exposes the stored bit so PEs can be chained as a simple shift
 * path during weight_load.
 */

`default_nettype none

module w1a8_pe #(
    parameter ACT_WIDTH  = 8,
    parameter PSUM_WIDTH = 24
) (
    input  wire                              clk,
    input  wire                              rst_n,

    input  wire                              clear,
    input  wire                              weight_load,
    input  wire                              weight_in,
    output wire                              weight_out,

    input  wire signed [ACT_WIDTH-1:0]       act_in,
    input  wire signed [PSUM_WIDTH-1:0]      psum_in,
    input  wire                              valid_in,

    output logic signed [ACT_WIDTH-1:0]      act_out,
    output logic signed [PSUM_WIDTH-1:0]     psum_out,
    output logic                             valid_out
);

    localparam CONTRIB_WIDTH = ACT_WIDTH + 1;

    logic weight_q;
    wire signed [CONTRIB_WIDTH-1:0] act_ext;
    wire signed [CONTRIB_WIDTH-1:0] contrib;
    wire signed [PSUM_WIDTH-1:0] contrib_ext;

    assign weight_out = weight_q;
    assign act_ext     = {act_in[ACT_WIDTH-1], act_in};
    assign contrib     = weight_q ? act_ext : -act_ext;
    assign contrib_ext = {{(PSUM_WIDTH-CONTRIB_WIDTH){contrib[CONTRIB_WIDTH-1]}}, contrib};

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight_q  <= 1'b0;
            act_out   <= {ACT_WIDTH{1'b0}};
            psum_out  <= {PSUM_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (clear) begin
            act_out   <= {ACT_WIDTH{1'b0}};
            psum_out  <= {PSUM_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (weight_load) begin
            weight_q  <= weight_in;
            act_out   <= {ACT_WIDTH{1'b0}};
            psum_out  <= {PSUM_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else begin
            act_out   <= act_in;
            psum_out  <= valid_in ? psum_in + contrib_ext : {PSUM_WIDTH{1'b0}};
            valid_out <= valid_in;
        end
    end

endmodule
