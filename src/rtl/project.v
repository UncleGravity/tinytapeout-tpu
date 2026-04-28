/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout wrapper for a parameterized W1A8 systolic tile.
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
 *             (assumes COLS <= 8; one LDW call per row)
 *   3 LDA     write one int8 activation (uio_in) to act_mem[arg_col]
 *   4 SEED    write one byte (uio_in) to acc_q[arg_row][arg_byte*8 +: 8]
 *   5 START   load stored weights into PEs, run one compute
 *   6 RDP     uo_out = acc_q[arg_row][arg_byte*8 +: 8]
 *   7 NOP     uo_out = status
 *
 * Argument layout:
 *   LDW       arg[ROW_SEL_WIDTH-1:0]                        = row
 *   LDA       arg[LDA_COL_WIDTH-1:0]                        = col
 *   SEED/RDP  arg[ROW_SEL_WIDTH-1:0]                        = row
 *             arg[ROW_SEL_WIDTH +: BYTE_SEL_WIDTH]          = byte
 *
 * Per-row accumulator (acc_q):
 *   One physical register per row holds the partial-sum accumulator. It is
 *   the seed (input) AND the result (output) of one compute, because those
 *   roles never overlap in time:
 *
 *     IDLE             host SEED writes  acc_q[r] = initial value
 *     RUN, cycle r     array READS       psum_in = acc_q[r]   (seed consumed)
 *     RUN, cycle r+COLS array WRITES     acc_q[r] = computed psum
 *     IDLE             host RDP reads    final value
 *
 *   Chaining: the host can RDP one run's accumulator and SEED it into the
 *   next run, which is how dot products longer than COLS are tiled (see
 *   replay_q8_block in test/bonsai_fixture.py).
 *
 * Status byte:
 *   [0] busy
 *   [1] done_latched
 *   [2] weight_load_done_latched
 *   [3] all result rows valid
 *   [4] start_ready (== idle)
 *   [5] reserved (held high while idle for protocol stability)
 *   [6] error_latched
 *   [7] reserved
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
    localparam integer ACT_WIDTH         = 8;
    localparam integer PSUM_WIDTH        = 16;
    localparam integer ROWS              = 2;
    localparam integer COLS              = 2;
    localparam integer PSUM_BYTES        = (PSUM_WIDTH + 7) / 8;
    localparam integer ROW_SEL_WIDTH     = (ROWS  <= 1) ? 1 : $clog2(ROWS);
    localparam integer BYTE_SEL_WIDTH    = (PSUM_BYTES <= 1) ? 1 : $clog2(PSUM_BYTES);
    localparam integer LDA_COL_WIDTH     = (COLS  <= 1) ? 1 : $clog2(COLS);
    localparam integer LOAD_COL_WIDTH    = LDA_COL_WIDTH;
    localparam integer COMPUTE_LAST_STEP = ROWS + COLS - 1;
    localparam integer STEP_MAX          = (COMPUTE_LAST_STEP > (COLS - 1))
                                                ? COMPUTE_LAST_STEP : (COLS - 1);
    localparam integer STEP_WIDTH        = (STEP_MAX < 1) ? 1 : $clog2(STEP_MAX + 1);

    // ------------------------------------------------------------------------
    // Command encoding
    // ------------------------------------------------------------------------
    localparam CMD_STATUS = 3'd0;
    localparam CMD_CLEAR  = 3'd1;
    localparam CMD_LDW    = 3'd2;
    localparam CMD_LDA    = 3'd3;
    localparam CMD_SEED   = 3'd4;
    localparam CMD_START  = 3'd5;
    localparam CMD_RDP    = 3'd6;
    localparam CMD_NOP    = 3'd7;

    localparam STATE_IDLE = 2'd0;
    localparam STATE_LOAD = 2'd1;
    localparam STATE_RUN  = 2'd2;

    wire [2:0] cmd = ui_in[2:0];
    wire [4:0] arg = ui_in[7:3];

    wire cmd_status = (cmd == CMD_STATUS);
    wire cmd_clear  = (cmd == CMD_CLEAR);
    wire cmd_ldw    = (cmd == CMD_LDW);
    wire cmd_lda    = (cmd == CMD_LDA);
    wire cmd_seed   = (cmd == CMD_SEED);
    wire cmd_start  = (cmd == CMD_START);
    wire cmd_rdp    = (cmd == CMD_RDP);
    wire cmd_nop    = (cmd == CMD_NOP);

    wire [ROW_SEL_WIDTH-1:0]  arg_row  = arg[ROW_SEL_WIDTH-1:0];
    wire [BYTE_SEL_WIDTH-1:0] arg_byte = arg[ROW_SEL_WIDTH +: BYTE_SEL_WIDTH];
    wire [LDA_COL_WIDTH-1:0]  arg_col  = arg[LDA_COL_WIDTH-1:0];

    // ------------------------------------------------------------------------
    // State + scratchpad
    // ------------------------------------------------------------------------
    logic [1:0]            state_q;
    logic [STEP_WIDTH-1:0] step_q;

    logic                            weight_mem [0:ROWS-1][0:COLS-1];
    logic signed [ACT_WIDTH-1:0]     act_mem    [0:COLS-1];
    logic signed [ROWS*PSUM_WIDTH-1:0] acc_q;
    logic [ROWS-1:0]                   acc_done_q;

    logic done_latched;
    logic weight_done_latched;
    logic error_latched;

    wire idle       = (state_q == STATE_IDLE);
    wire load_phase = (state_q == STATE_LOAD);
    wire run_phase  = (state_q == STATE_RUN);
    wire busy       = !idle;

    // ------------------------------------------------------------------------
    // Output muxes
    // ------------------------------------------------------------------------
    wire [7:0] selected_result_byte =
        (arg_row < ROWS) ?
            acc_q[arg_row*PSUM_WIDTH + arg_byte*8 +: 8] :
            8'h00;

    wire [7:0] status_byte = {
        1'b0,
        error_latched,
        idle,
        idle,
        &acc_done_q,
        weight_done_latched,
        done_latched,
        busy
    };

    assign uo_out  = cmd_rdp ? selected_result_byte : status_byte;
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // ------------------------------------------------------------------------
    // Compute datapath: drives the systolic array directly.
    //
    // STATE_LOAD: step_q = 0..COLS-1 counts cycles. Weight column COLS-1 is
    //             shifted in first (it must reach the rightmost PE), so
    //             load_col_idx walks COLS-1 down to 0 as step_q walks up.
    // STATE_RUN:  column c receives its activation at step_q == c (column
    //             skew); row r receives its valid+seed pulse at step_q == r
    //             (row skew). The array's per-row valid_out chain marks each
    //             result row's ready cycle; we latch psum then.
    // ------------------------------------------------------------------------

    wire [LOAD_COL_WIDTH-1:0] load_col_idx =
        (COLS - 1) - step_q[LOAD_COL_WIDTH-1:0];

    logic [ROWS-1:0] array_weight_in;
    always_comb begin
        array_weight_in = {ROWS{1'b0}};
        for (int r = 0; r < ROWS; r = r + 1) begin
            array_weight_in[r] = weight_mem[r][load_col_idx];
        end
    end

    wire signed [COLS*ACT_WIDTH-1:0]  array_act_in;
    wire signed [COLS*ACT_WIDTH-1:0]  array_act_out;
    wire signed [ROWS*PSUM_WIDTH-1:0] array_psum_in;
    wire signed [ROWS*PSUM_WIDTH-1:0] array_psum_out;
    wire [ROWS-1:0]                   array_valid_in;
    wire [ROWS-1:0]                   array_valid_out;

    genvar gc;
    generate
        for (gc = 0; gc < COLS; gc = gc + 1) begin : gen_act_skew
            assign array_act_in[gc*ACT_WIDTH +: ACT_WIDTH] =
                (run_phase && (step_q == gc)) ?
                    act_mem[gc] : {ACT_WIDTH{1'b0}};
        end
    endgenerate

    genvar gr;
    generate
        for (gr = 0; gr < ROWS; gr = gr + 1) begin : gen_psum_skew
            assign array_valid_in[gr] = run_phase && (step_q == gr);
            assign array_psum_in[gr*PSUM_WIDTH +: PSUM_WIDTH] =
                (run_phase && (step_q == gr)) ?
                    acc_q[gr*PSUM_WIDTH +: PSUM_WIDTH] :
                    {PSUM_WIDTH{1'b0}};
        end
    endgenerate

    w1a8_array #(
        .ACT_WIDTH (ACT_WIDTH),
        .PSUM_WIDTH(PSUM_WIDTH),
        .ROWS      (ROWS),
        .COLS      (COLS)
    ) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear      (cmd_clear),
        .weight_load(load_phase),
        .weight_in  (array_weight_in),
        .act_in     (array_act_in),
        .act_out    (array_act_out),
        .psum_in    (array_psum_in),
        .valid_in   (array_valid_in),
        .psum_out   (array_psum_out),
        .valid_out  (array_valid_out)
    );

    // ------------------------------------------------------------------------
    // Single sequential block: FSM, scratchpad, accumulator.
    //
    // acc_q is the per-row accumulator: SEED writes (in IDLE) load the
    // starting value; the array overwrites it later in RUN with the computed
    // psum. The two writes are mutually exclusive in time, so no conflict.
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state_q             <= STATE_IDLE;
            step_q              <= {STEP_WIDTH{1'b0}};
            done_latched        <= 1'b0;
            weight_done_latched <= 1'b0;
            error_latched       <= 1'b0;
            acc_q            <= {ROWS*PSUM_WIDTH{1'b0}};
            acc_done_q      <= {ROWS{1'b0}};
            for (int row_i = 0; row_i < ROWS; row_i = row_i + 1) begin
                for (int col_i = 0; col_i < COLS; col_i = col_i + 1) begin
                    weight_mem[row_i][col_i] <= 1'b0;
                end
            end
            for (int col_i = 0; col_i < COLS; col_i = col_i + 1) begin
                act_mem[col_i] <= {ACT_WIDTH{1'b0}};
            end
        end else if (cmd_clear) begin
            state_q             <= STATE_IDLE;
            step_q              <= {STEP_WIDTH{1'b0}};
            done_latched        <= 1'b0;
            weight_done_latched <= 1'b0;
            error_latched       <= 1'b0;
            acc_q            <= {ROWS*PSUM_WIDTH{1'b0}};
            acc_done_q      <= {ROWS{1'b0}};
            for (int col_i = 0; col_i < COLS; col_i = col_i + 1) begin
                act_mem[col_i] <= {ACT_WIDTH{1'b0}};
            end
        end else begin
            // Capture each row's psum on the cycle its valid_out fires.
            // Only fires during/just-after RUN; in IDLE the array is quiet.
            for (int r = 0; r < ROWS; r = r + 1) begin
                if (array_valid_out[r]) begin
                    acc_q[r*PSUM_WIDTH +: PSUM_WIDTH] <=
                        array_psum_out[r*PSUM_WIDTH +: PSUM_WIDTH];
                    acc_done_q[r] <= 1'b1;
                end
            end

            case (state_q)
                STATE_IDLE: begin
                    if (cmd_ldw) begin
                        if (arg_row < ROWS) begin
                            for (int i = 0; i < COLS; i = i + 1) begin
                                weight_mem[arg_row][i] <= uio_in[i];
                            end
                        end else begin
                            error_latched <= 1'b1;
                        end
                    end

                    if (cmd_lda) begin
                        if (arg_col < COLS) begin
                            act_mem[arg_col] <= uio_in;
                        end else begin
                            error_latched <= 1'b1;
                        end
                    end

                    if (cmd_seed) begin
                        if (arg_row < ROWS) begin
                            acc_q[arg_row*PSUM_WIDTH + arg_byte*8 +: 8] <= uio_in;
                        end else begin
                            error_latched <= 1'b1;
                        end
                    end

                    if (cmd_start) begin
                        state_q             <= STATE_LOAD;
                        step_q              <= {STEP_WIDTH{1'b0}};
                        done_latched        <= 1'b0;
                        weight_done_latched <= 1'b0;
                        acc_done_q      <= {ROWS{1'b0}};
                    end
                end

                STATE_LOAD: begin
                    if (step_q == COLS - 1) begin
                        state_q             <= STATE_RUN;
                        step_q              <= {STEP_WIDTH{1'b0}};
                        weight_done_latched <= 1'b1;
                    end else begin
                        step_q <= step_q + 1'b1;
                    end
                end

                STATE_RUN: begin
                    if (step_q < COMPUTE_LAST_STEP[STEP_WIDTH-1:0]) begin
                        step_q <= step_q + 1'b1;
                    end
                    if ((step_q >= COMPUTE_LAST_STEP[STEP_WIDTH-1:0]) &&
                        (&acc_done_q)) begin
                        state_q      <= STATE_IDLE;
                        done_latched <= 1'b1;
                    end
                end

                default: begin
                    state_q <= STATE_IDLE;
                end
            endcase
        end
    end

    wire _unused = &{ena, cmd_status, cmd_nop, array_act_out, 1'b0};

endmodule
