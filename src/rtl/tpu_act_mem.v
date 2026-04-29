/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_act_mem - per-column int8 activation scratchpad.
 *
 * Host write port (LDA):
 *   `we` writes wdata into act_mem[arg[COL_SEL_W-1:0]].
 *   Out-of-range column selectors pulse `err`.
 *
 * Array drive port:
 *   During run_phase, column gc gets its activation when step == gc; outside
 *   that window the bus is zero. This implements the per-column skew needed
 *   by the systolic wavefront.
 *
 * Both async reset and `clear` zero the cells (clear is a soft reset that
 * also resets the FSM and accumulator).
 */

`default_nettype none

module tpu_act_mem #(
    parameter ROWS      = 2,
    parameter COLS      = 2,
    parameter ACT_WIDTH = 8
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,

    // host write (LDA), gated upstream to fire only in IDLE
    input  wire                              we,
    input  wire [4:0]                        arg,
    input  wire [7:0]                        wdata,
    output wire                              err,

    // array drive
    input  wire                              run_phase,
    input  wire [$clog2(ROWS+COLS)-1:0]      step,
    output wire signed [COLS*ACT_WIDTH-1:0]  array_act_in
);

    localparam COL_SEL_W = (COLS <= 1) ? 1 : $clog2(COLS);

    logic signed [ACT_WIDTH-1:0] act_mem [0:COLS-1];

    wire [COL_SEL_W-1:0] wcol    = arg[COL_SEL_W-1:0];
    wire                 col_oob = (wcol >= COLS);

    assign err = we && col_oob;

    integer c;
    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            for (c = 0; c < COLS; c = c + 1)
                act_mem[c] <= {ACT_WIDTH{1'b0}};
        end else if (we && !col_oob) begin
            act_mem[wcol] <= wdata;
        end
    end

    genvar gc;
    generate
        for (gc = 0; gc < COLS; gc = gc + 1) begin : g_act_skew
            assign array_act_in[gc*ACT_WIDTH +: ACT_WIDTH] =
                (run_phase && (step == gc)) ?
                    act_mem[gc] : {ACT_WIDTH{1'b0}};
        end
    endgenerate

endmodule
