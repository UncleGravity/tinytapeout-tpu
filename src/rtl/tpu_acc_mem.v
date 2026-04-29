/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_acc_mem - per-row partial-sum accumulator file.
 *
 * One PSUM_WIDTH register per row holds both the seed (input to a run) and
 * the result (output of a run). The two roles never overlap in time:
 *
 *   IDLE                   host SEED writes  acc_q[r] = initial value
 *   RUN, step == r         seed consumed     array reads acc_q[r]
 *   RUN, ~COLS later       array writes      acc_q[r] = computed psum
 *   IDLE                   host RDP reads    final value
 *
 * Per-row done flags arm on capture and reset on start_pulse so the next run
 * can detect "all rows valid" cleanly.
 *
 * Reset behavior:
 *   !rst_n or clear -> zero acc_q and acc_done_q
 *   start_pulse     -> zero acc_done_q only (preserve seeded acc_q)
 */

`default_nettype none

module tpu_acc_mem #(
    parameter ROWS       = 2,
    parameter COLS       = 2,
    parameter PSUM_WIDTH = 16
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,

    // host SEED write, gated upstream to fire only in IDLE
    input  wire                              we_seed,
    input  wire [4:0]                        arg,
    input  wire [7:0]                        wdata,
    output wire                              err,

    // per-row capture from the array
    input  wire [ROWS-1:0]                   capture_valid,
    input  wire signed [ROWS*PSUM_WIDTH-1:0] capture_psum,

    // arms acc_done_q for the upcoming run
    input  wire                              start_pulse,

    // array drive: row gr seeds psum at step == gr during run_phase
    input  wire                              run_phase,
    input  wire [$clog2(ROWS+COLS)-1:0]      step,
    output wire signed [ROWS*PSUM_WIDTH-1:0] array_psum_in,
    output wire [ROWS-1:0]                   array_valid_in,

    // host read (RDP) and run-completion summary
    output wire [7:0]                        result_byte,
    output wire                              all_rows_done
);

    localparam PSUM_BYTES = (PSUM_WIDTH + 7) / 8;
    localparam ROW_SEL_W  = (ROWS <= 1) ? 1 : $clog2(ROWS);
    localparam BYTE_SEL_W = (PSUM_BYTES <= 1) ? 1 : $clog2(PSUM_BYTES);

    logic signed [ROWS*PSUM_WIDTH-1:0] acc_q;
    logic [ROWS-1:0]                   acc_done_q;

    wire [ROW_SEL_W-1:0]  arg_row  = arg[ROW_SEL_W-1:0];
    wire [BYTE_SEL_W-1:0] arg_byte = arg[ROW_SEL_W +: BYTE_SEL_W];
    wire                  row_oob  = (arg_row >= ROWS);

    assign err           = we_seed && row_oob;
    assign all_rows_done = &acc_done_q;
    assign result_byte   = row_oob ? 8'h00 :
        acc_q[arg_row*PSUM_WIDTH + arg_byte*8 +: 8];

    integer r;
    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            acc_q      <= {ROWS*PSUM_WIDTH{1'b0}};
            acc_done_q <= {ROWS{1'b0}};
        end else begin
            // Capture the array's psum on the cycle each row's wavefront exits.
            for (r = 0; r < ROWS; r = r + 1) begin
                if (capture_valid[r]) begin
                    acc_q[r*PSUM_WIDTH +: PSUM_WIDTH] <=
                        capture_psum[r*PSUM_WIDTH +: PSUM_WIDTH];
                    acc_done_q[r] <= 1'b1;
                end
            end

            // Host SEED takes precedence (different cycle in practice; safe by ordering).
            if (we_seed && !row_oob) begin
                acc_q[arg_row*PSUM_WIDTH + arg_byte*8 +: 8] <= wdata;
            end

            if (start_pulse) begin
                acc_done_q <= {ROWS{1'b0}};
            end
        end
    end

    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr = gr + 1) begin : g_psum_skew
            assign array_valid_in[gr] = run_phase && (step == gr);
            assign array_psum_in[gr*PSUM_WIDTH +: PSUM_WIDTH] =
                (run_phase && (step == gr)) ?
                    acc_q[gr*PSUM_WIDTH +: PSUM_WIDTH] :
                    {PSUM_WIDTH{1'b0}};
        end
    endgenerate

endmodule
