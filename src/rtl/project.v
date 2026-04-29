/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout top wrapper for a parameterized W1A8 systolic tile.
 *
 * Block diagram:
 *
 *   ui_in --+-> tpu_cmd_decode --+-> cmd_*, arg
 *           |                    +-> tpu_tile_ctrl  (FSM, step counter)
 *           |                    +-> tpu_weight_mem (LDW + shift drive)
 *   uio_in -+------------------- +-> tpu_act_mem    (LDA + skew drive)
 *                                +-> tpu_acc_mem    (SEED + capture + RDP)
 *                                |
 *                                +-> w1a8_array     (PE/row/array)
 *                                |
 *           uo_out <-- tpu_status (status byte / RDP result mux)
 *
 * Pin protocol:
 *   ui_in[2:0] command
 *   ui_in[7:3] argument (5 bits, layout depends on command)
 *   uio_in     data byte
 *   uo_out     status byte, or RDP result byte
 *
 * Commands:
 *   0 STATUS  uo_out = status
 *   1 CLEAR   reset FSM, clear acts/results, keep weights in PEs
 *   2 LDW     write COLS packed weight bits (uio_in[COLS-1:0]) to row arg_row
 *   3 LDA     write one int8 activation (uio_in) to act_mem[arg_col]
 *   4 SEED    write one byte (uio_in) to acc_q[arg_row][arg_byte*8 +: 8]
 *   5 START   load stored weights into PEs, run one compute
 *   6 RDP     uo_out = acc_q[arg_row][arg_byte*8 +: 8]
 *   7 NOP     uo_out = status
 *
 * Status byte: see tpu_status.v.
 *
 * Per-run accumulator (acc_q): see tpu_acc_mem.v. The host can RDP one run's
 * accumulator and SEED it into the next, which is how dot products longer
 * than COLS are tiled (see replay_q8_block in test/bonsai_fixture.py).
 */

`default_nettype none

module tt_um_unclegravity_tpu (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    // ------------------------------------------------------------------------
    // Tile parameters
    // ------------------------------------------------------------------------
    localparam ACT_WIDTH  = 8;
    localparam PSUM_WIDTH = 16;
    localparam ROWS       = 2;
    localparam COLS       = 2;

    localparam STEP_W = $clog2(ROWS + COLS);

    // ------------------------------------------------------------------------
    // Command decode
    // ------------------------------------------------------------------------
    wire       cmd_clear, cmd_ldw, cmd_lda, cmd_seed, cmd_start, cmd_rdp;
    wire [4:0] arg;

    tpu_cmd_decode u_decode (
        // inputs
        .ui_in     (ui_in),
        // outputs
        .cmd_clear (cmd_clear),
        .cmd_ldw   (cmd_ldw),
        .cmd_lda   (cmd_lda),
        .cmd_seed  (cmd_seed),
        .cmd_start (cmd_start),
        .cmd_rdp   (cmd_rdp),
        .arg       (arg)
    );

    // ------------------------------------------------------------------------
    // FSM and step counter
    // ------------------------------------------------------------------------
    wire              idle, busy, load_phase, run_phase;
    wire [STEP_W-1:0] step;
    wire              start_pulse, done_latched, weight_done_latched;
    wire              all_rows_done;

    tpu_tile_ctrl #(
        .ROWS(ROWS),
        .COLS(COLS)
    ) u_ctrl (
        // inputs
        .clk                 (clk),
        .rst_n               (rst_n),
        .clear               (cmd_clear),
        .cmd_start           (cmd_start),
        .all_rows_done       (all_rows_done),
        // outputs
        .idle                (idle),
        .busy                (busy),
        .load_phase          (load_phase),
        .run_phase           (run_phase),
        .step                (step),
        .start_pulse         (start_pulse),
        .done_latched        (done_latched),
        .weight_done_latched (weight_done_latched)
    );

    // ------------------------------------------------------------------------
    // Scratchpads (each owns its own host-write port + array drive logic)
    // ------------------------------------------------------------------------
    wire [ROWS-1:0]                   array_weight_in;
    wire signed [COLS*ACT_WIDTH-1:0]  array_act_in;
    wire signed [ROWS*PSUM_WIDTH-1:0] array_psum_in;
    wire [ROWS-1:0]                   array_valid_in;

    wire weight_err, act_err, acc_err;
    wire [7:0] result_byte;

    tpu_weight_mem #(
        .ROWS(ROWS),
        .COLS(COLS)
    ) u_wmem (
        // inputs
        .clk            (clk),
        .rst_n          (rst_n),
        .we             (cmd_ldw && idle),
        .arg            (arg),
        .wdata          (uio_in),
        .weight_load    (load_phase),
        .step           (step),
        // outputs
        .err            (weight_err),
        .weight_col_out (array_weight_in)
    );

    tpu_act_mem #(
        .ROWS      (ROWS),
        .COLS      (COLS),
        .ACT_WIDTH (ACT_WIDTH)
    ) u_amem (
        // inputs
        .clk          (clk),
        .rst_n        (rst_n),
        .clear        (cmd_clear),
        .we           (cmd_lda && idle),
        .arg          (arg),
        .wdata        (uio_in),
        .run_phase    (run_phase),
        .step         (step),
        // outputs
        .err          (act_err),
        .array_act_in (array_act_in)
    );

    // ------------------------------------------------------------------------
    // Compute fabric
    // ------------------------------------------------------------------------
    wire signed [COLS*ACT_WIDTH-1:0]  array_act_out;
    wire signed [ROWS*PSUM_WIDTH-1:0] array_psum_out;
    wire [ROWS-1:0]                   array_valid_out;

    w1a8_array #(
        .ACT_WIDTH (ACT_WIDTH),
        .PSUM_WIDTH(PSUM_WIDTH),
        .ROWS      (ROWS),
        .COLS      (COLS)
    ) u_array (
        // inputs
        .clk         (clk),
        .rst_n       (rst_n),
        .clear       (cmd_clear),
        .weight_load (load_phase),
        .weight_in   (array_weight_in),
        .act_in      (array_act_in),
        .psum_in     (array_psum_in),
        .valid_in    (array_valid_in),
        // outputs
        .act_out     (array_act_out),
        .psum_out    (array_psum_out),
        .valid_out   (array_valid_out)
    );

    tpu_acc_mem #(
        .ROWS       (ROWS),
        .COLS       (COLS),
        .PSUM_WIDTH (PSUM_WIDTH)
    ) u_accmem (
        // inputs
        .clk            (clk),
        .rst_n          (rst_n),
        .clear          (cmd_clear),
        .we_seed        (cmd_seed && idle),
        .arg            (arg),
        .wdata          (uio_in),
        .capture_valid  (array_valid_out),
        .capture_psum   (array_psum_out),
        .start_pulse    (start_pulse),
        .run_phase      (run_phase),
        .step           (step),
        // outputs
        .err            (acc_err),
        .array_psum_in  (array_psum_in),
        .array_valid_in (array_valid_in),
        .result_byte    (result_byte),
        .all_rows_done  (all_rows_done)
    );

    // ------------------------------------------------------------------------
    // Sticky error flag (OR-reduce of per-memory bounds-check pulses)
    // ------------------------------------------------------------------------
    logic error_latched;
    always_ff @(posedge clk) begin
        if (!rst_n || cmd_clear) begin
            error_latched <= 1'b0;
        end else if (weight_err || act_err || acc_err) begin
            error_latched <= 1'b1;
        end
    end

    // ------------------------------------------------------------------------
    // Output mux
    // ------------------------------------------------------------------------
    tpu_status u_status (
        // inputs
        .cmd_rdp             (cmd_rdp),
        .result_byte         (result_byte),
        .busy                (busy),
        .idle                (idle),
        .done_latched        (done_latched),
        .weight_done_latched (weight_done_latched),
        .all_rows_done       (all_rows_done),
        .error_latched       (error_latched),
        // outputs
        .uo_out              (uo_out)
    );

    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // Tied / unused inputs to keep lint quiet.
    wire _unused = &{ena, array_act_out, 1'b0};

endmodule
