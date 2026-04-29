/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_weight_mem - ROWS x COLS bit-cell weight scratchpad.
 *
 * Host write port (LDW):
 *   `we` writes wdata[COLS-1:0] into row arg[ROW_SEL_W-1:0].
 *   Out-of-range row selectors are rejected and pulse `err`.
 *
 * Array drive port (during weight_load):
 *   step (0..COLS-1) walks COLS-1 down to 0 so column COLS-1 is shifted in
 *   first, matching the row's serial weight pipeline (the rightmost PE must
 *   end up holding column COLS-1's weight). When weight_load is low, the
 *   bus is forced to zero so the diagram and waveforms read clean.
 *
 * Reset preserves weights; only async reset zeros the cells.
 */

`default_nettype none

module tpu_weight_mem #(
    parameter ROWS = 2,
    parameter COLS = 2
) (
    input  wire                              clk,
    input  wire                              rst_n,

    // host write (LDW), gated upstream to fire only in IDLE
    input  wire                              we,
    input  wire [4:0]                        arg,
    input  wire [7:0]                        wdata,
    output wire                              err,

    // array drive: during weight_load, present col (COLS-1)-step on every row
    input  wire                              weight_load,
    input  wire [$clog2(ROWS+COLS)-1:0]      step,
    output wire [ROWS-1:0]                   weight_col_out
);

    localparam ROW_SEL_W = (ROWS <= 1) ? 1 : $clog2(ROWS);
    localparam COL_SEL_W = (COLS <= 1) ? 1 : $clog2(COLS);

    logic weight_mem [0:ROWS-1][0:COLS-1];

    wire [ROW_SEL_W-1:0] wrow    = arg[ROW_SEL_W-1:0];
    wire                 row_oob = (wrow >= ROWS);

    assign err = we && row_oob;

    integer r, c;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLS; c = c + 1)
                    weight_mem[r][c] <= 1'b0;
        end else if (we && !row_oob) begin
            for (c = 0; c < COLS; c = c + 1)
                weight_mem[wrow][c] <= wdata[c];
        end
    end

    wire [COL_SEL_W-1:0] read_col = (COLS-1) - step[COL_SEL_W-1:0];

    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr = gr + 1) begin : g_wcol
            assign weight_col_out[gr] =
                weight_load ? weight_mem[gr][read_col] : 1'b0;
        end
    endgenerate

endmodule
