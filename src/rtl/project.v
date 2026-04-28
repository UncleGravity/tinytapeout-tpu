/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout wrapper for a parameterized W1A8 systolic tile.
 *
 * Pin protocol:
 *   ui_in[2:0] command
 *   ui_in[7:3] argument
 *   uio_in     data byte
 *   uo_out     status or read data
 *
 * Commands:
 *   0 STATUS    uo_out shows status
 *   1 CLEAR     clear FSM/datapath state, keep loaded weights
 *   2 SET_ADDR  arg[1:0] selects address register, uio_in is value
 *   3 WRITE     write uio_in to selected bank/address
 *   4 READ      uo_out reads selected bank/address
 *   5 START     load stored weights into PEs and run one compute
 *   7 NOP       uo_out shows status
 *
 * Address registers:
 *   0 row_addr   output row / weight row
 *   1 col_addr   activation column / first packed weight column
 *   2 byte_addr  seed/result byte (0 = LSB, 1 = MSB)
 *   3 bank_addr  selected bank
 *
 * Banks:
 *   1 WEIGHT   WRITE packs up to 8 one-bit weights starting at col_addr
 *   2 ACT      WRITE one int8 activation at col_addr
 *   3 SEED     WRITE seed byte at row_addr/byte_addr
 *   4 RESULT   READ result byte at row_addr/byte_addr
 *   5 STATUS   READ status byte
 *
 * Status byte:
 *   [0] busy
 *   [1] done_latched
 *   [2] weight_load_done_latched
 *   [3] all result rows valid
 *   [4] start_ready (== idle)
 *   [5] weight_load_ready (== idle)
 *   [6] error_latched
 *   [7] reserved
 *
 * Compute phases (after START):
 *   LOAD : COLS cycles, shifts weight_mem rows into PE chains, last col first
 *   RUN  : drives column-skewed activations and row-skewed valid+seed pulses
 *          into the array; captures each row's psum on the cycle its
 *          valid_out fires.
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

    localparam integer ACT_WIDTH         = 8;
    localparam integer PSUM_WIDTH        = 16;
    localparam integer ROWS              = 2;
    localparam integer COLS              = 4;
    localparam integer PSUM_BYTES        = (PSUM_WIDTH + 7) / 8;
    localparam integer ROW_ADDR_WIDTH    = 4;
    localparam integer COL_ADDR_WIDTH    = 4;
    localparam integer BYTE_ADDR_WIDTH   = (PSUM_BYTES <= 1) ? 1 : $clog2(PSUM_BYTES);
    localparam integer BANK_WIDTH        = 3;
    localparam integer LOAD_COL_WIDTH    = (COLS <= 1) ? 1 : $clog2(COLS);
    localparam integer COMPUTE_LAST_STEP = ROWS + COLS - 1;
    localparam integer STEP_MAX          = (COMPUTE_LAST_STEP > (COLS - 1))
                                                ? COMPUTE_LAST_STEP : (COLS - 1);
    localparam integer STEP_WIDTH        = (STEP_MAX < 1) ? 1 : $clog2(STEP_MAX + 1);

    localparam CMD_STATUS   = 3'd0;
    localparam CMD_CLEAR    = 3'd1;
    localparam CMD_SET_ADDR = 3'd2;
    localparam CMD_WRITE    = 3'd3;
    localparam CMD_READ     = 3'd4;
    localparam CMD_START    = 3'd5;
    localparam CMD_NOP      = 3'd7;

    localparam ADDR_ROW  = 2'd0;
    localparam ADDR_COL  = 2'd1;
    localparam ADDR_BYTE = 2'd2;
    localparam ADDR_BANK = 2'd3;

    localparam BANK_WEIGHT = 3'd1;
    localparam BANK_ACT    = 3'd2;
    localparam BANK_SEED   = 3'd3;
    localparam BANK_RESULT = 3'd4;
    localparam BANK_STATUS = 3'd5;

    localparam STATE_IDLE = 2'd0;
    localparam STATE_LOAD = 2'd1;
    localparam STATE_RUN  = 2'd2;

    wire [2:0] cmd = ui_in[2:0];
    wire [4:0] arg = ui_in[7:3];

    wire cmd_status   = (cmd == CMD_STATUS);
    wire cmd_clear    = (cmd == CMD_CLEAR);
    wire cmd_set_addr = (cmd == CMD_SET_ADDR);
    wire cmd_write    = (cmd == CMD_WRITE);
    wire cmd_read     = (cmd == CMD_READ);
    wire cmd_start    = (cmd == CMD_START);
    wire cmd_nop      = (cmd == CMD_NOP);

    logic [ROW_ADDR_WIDTH-1:0]  row_addr;
    logic [COL_ADDR_WIDTH-1:0]  col_addr;
    logic [BYTE_ADDR_WIDTH-1:0] byte_addr;
    logic [BANK_WIDTH-1:0]      bank_addr;

    logic [1:0]              state_q;
    logic [STEP_WIDTH-1:0]   step_q;

    logic                            weight_mem [0:ROWS-1][0:COLS-1];
    logic signed [ACT_WIDTH-1:0]     act_mem    [0:COLS-1];
    logic signed [PSUM_WIDTH-1:0]    seed_mem   [0:ROWS-1];

    logic signed [ROWS*PSUM_WIDTH-1:0] result_q;
    logic [ROWS-1:0]                   result_valid_q;

    logic done_latched;
    logic weight_done_latched;
    logic error_latched;

    wire idle       = (state_q == STATE_IDLE);
    wire load_phase = (state_q == STATE_LOAD);
    wire run_phase  = (state_q == STATE_RUN);
    wire busy       = !idle;

    wire signed [PSUM_WIDTH-1:0] selected_result =
        (row_addr < ROWS) ?
            result_q[row_addr*PSUM_WIDTH +: PSUM_WIDTH] :
            {PSUM_WIDTH{1'b0}};
    wire signed [PSUM_WIDTH-1:0] selected_seed =
        (row_addr < ROWS) ? seed_mem[row_addr] : {PSUM_WIDTH{1'b0}};

    wire [7:0] selected_result_byte = selected_result[byte_addr*8 +: 8];
    wire [7:0] selected_seed_byte   = selected_seed  [byte_addr*8 +: 8];

    wire [7:0] status_byte = {
        1'b0,
        error_latched,
        idle,
        idle,
        &result_valid_q,
        weight_done_latched,
        done_latched,
        busy
    };

    logic [7:0] read_data;
    always_comb begin
        case (bank_addr)
            BANK_ACT:    read_data = (col_addr < COLS) ? act_mem[col_addr] : 8'h00;
            BANK_SEED:   read_data = (row_addr < ROWS) ? selected_seed_byte : 8'h00;
            BANK_RESULT: read_data = selected_result_byte;
            default:     read_data = status_byte;
        endcase
    end

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
                    seed_mem[gr] : {PSUM_WIDTH{1'b0}};
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
    // Result capture: latch each row's psum on the cycle its valid_out fires.
    // Cleared on reset, CLEAR, and at the start of a new compute.
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n || cmd_clear || (idle && cmd_start)) begin
            result_q       <= {ROWS*PSUM_WIDTH{1'b0}};
            result_valid_q <= {ROWS{1'b0}};
        end else begin
            for (int r = 0; r < ROWS; r = r + 1) begin
                if (array_valid_out[r]) begin
                    result_q[r*PSUM_WIDTH +: PSUM_WIDTH] <=
                        array_psum_out[r*PSUM_WIDTH +: PSUM_WIDTH];
                    result_valid_q[r] <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // Host I/O FSM + scratchpad.
    // ------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            row_addr            <= {ROW_ADDR_WIDTH{1'b0}};
            col_addr            <= {COL_ADDR_WIDTH{1'b0}};
            byte_addr           <= {BYTE_ADDR_WIDTH{1'b0}};
            bank_addr           <= BANK_STATUS;
            state_q             <= STATE_IDLE;
            step_q              <= {STEP_WIDTH{1'b0}};
            done_latched        <= 1'b0;
            weight_done_latched <= 1'b0;
            error_latched       <= 1'b0;
            for (int row_i = 0; row_i < ROWS; row_i = row_i + 1) begin
                seed_mem[row_i] <= {PSUM_WIDTH{1'b0}};
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
            for (int row_i = 0; row_i < ROWS; row_i = row_i + 1) begin
                seed_mem[row_i] <= {PSUM_WIDTH{1'b0}};
            end
            for (int col_i = 0; col_i < COLS; col_i = col_i + 1) begin
                act_mem[col_i] <= {ACT_WIDTH{1'b0}};
            end
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (cmd_set_addr) begin
                        case (arg[1:0])
                            ADDR_ROW:  row_addr  <= uio_in[ROW_ADDR_WIDTH-1:0];
                            ADDR_COL:  col_addr  <= uio_in[COL_ADDR_WIDTH-1:0];
                            ADDR_BYTE: byte_addr <= uio_in[BYTE_ADDR_WIDTH-1:0];
                            default:   bank_addr <= uio_in[BANK_WIDTH-1:0];
                        endcase
                    end

                    if (cmd_write) begin
                        case (bank_addr)
                            BANK_WEIGHT: begin
                                if (row_addr < ROWS) begin
                                    for (int packed_col_i = 0; packed_col_i < 8; packed_col_i = packed_col_i + 1) begin
                                        if ((col_addr + packed_col_i) < COLS) begin
                                            weight_mem[row_addr][col_addr + packed_col_i] <= uio_in[packed_col_i];
                                        end
                                    end
                                end else begin
                                    error_latched <= 1'b1;
                                end
                            end
                            BANK_ACT: begin
                                if (col_addr < COLS) begin
                                    act_mem[col_addr] <= uio_in;
                                end else begin
                                    error_latched <= 1'b1;
                                end
                            end
                            BANK_SEED: begin
                                if (row_addr < ROWS) begin
                                    seed_mem[row_addr][byte_addr*8 +: 8] <= uio_in;
                                end else begin
                                    error_latched <= 1'b1;
                                end
                            end
                            default: begin
                            end
                        endcase
                    end

                    if (cmd_start) begin
                        done_latched        <= 1'b0;
                        weight_done_latched <= 1'b0;
                        state_q             <= STATE_LOAD;
                        step_q              <= {STEP_WIDTH{1'b0}};
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
                        (&result_valid_q)) begin
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

    assign uo_out  = cmd_read ? read_data : status_byte;
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    wire _unused = &{ena, cmd_status, cmd_nop, array_act_out, 1'b0};

endmodule
