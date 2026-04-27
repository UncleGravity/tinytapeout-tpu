/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyTapeout v1 wrapper for the W1A8 systolic tile.
 *
 * Fixed tile:
 *   ROWS = 2 output rows
 *   COLS = 4 activation/weight columns
 *
 * Pin protocol:
 *   ui_in[2:0] command
 *   ui_in[4:3] index     col index for LOAD_ACT, byte index for LOAD_SEED/READ
 *   ui_in[5]   row       row index for LOAD_SEED/READ
 *   ui_in[7:6] reserved
 *   uio_in     data byte / packed weight bits
 *   uo_out     status or selected result byte
 *
 * Commands:
 *   0 NOP          uo_out shows status
 *   1 CLEAR        clear wrapper/controller state
 *   2 LOAD_WEIGHT  uio_in[1:0] = one physical shift bit per row
 *                  Send COLS cycles, last column first.
 *   3 LOAD_ACT     ui index selects column, uio_in is int8 activation
 *   4 LOAD_SEED    ui row/index select seed byte, little-endian 24-bit
 *   5 START        start one activation vector transaction
 *   6 READ_RESULT  ui row/index select result byte; index 3 sign-extends
 *   7 STATUS       uo_out shows status
 *
 * Status byte:
 *   [0] busy
 *   [1] done_latched
 *   [2] weight_load_done_latched
 *   [3] all result rows valid
 *   [4] start_ready
 *   [5] weight_load_ready
 *   [6] result_valid[0]
 *   [7] result_valid[1]
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

    localparam CMD_NOP         = 3'd0;
    localparam CMD_CLEAR       = 3'd1;
    localparam CMD_LOAD_WEIGHT = 3'd2;
    localparam CMD_LOAD_ACT    = 3'd3;
    localparam CMD_LOAD_SEED   = 3'd4;
    localparam CMD_START       = 3'd5;
    localparam CMD_READ_RESULT = 3'd6;
    localparam CMD_STATUS      = 3'd7;

    wire [2:0] cmd        = ui_in[2:0];
    wire [1:0] index      = ui_in[4:3];
    wire       row_select = ui_in[5];

    wire cmd_clear       = (cmd == CMD_CLEAR);
    wire cmd_load_weight = (cmd == CMD_LOAD_WEIGHT);
    wire cmd_load_act    = (cmd == CMD_LOAD_ACT);
    wire cmd_load_seed   = (cmd == CMD_LOAD_SEED);
    wire cmd_start       = (cmd == CMD_START);
    wire cmd_read_result = (cmd == CMD_READ_RESULT);
    wire cmd_status      = (cmd == CMD_STATUS);

    logic signed [7:0]  act0;
    logic signed [7:0]  act1;
    logic signed [7:0]  act2;
    logic signed [7:0]  act3;
    logic signed [23:0] seed0;
    logic signed [23:0] seed1;

    wire signed [31:0] act_vector = {act3, act2, act1, act0};
    wire signed [47:0] seed_in    = {seed1, seed0};

    wire        controller_weight_done;
    wire        controller_start_ready;
    wire        controller_weight_ready;
    wire        controller_busy;
    wire        controller_done;
    wire [1:0]  controller_weight_out;
    wire [1:0]  controller_result_valid;
    wire signed [47:0] controller_result;

    logic done_latched;
    logic weight_done_latched;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            act0                <= 8'sd0;
            act1                <= 8'sd0;
            act2                <= 8'sd0;
            act3                <= 8'sd0;
            seed0               <= 24'sd0;
            seed1               <= 24'sd0;
            done_latched        <= 1'b0;
            weight_done_latched <= 1'b0;
        end else if (cmd_clear) begin
            act0                <= 8'sd0;
            act1                <= 8'sd0;
            act2                <= 8'sd0;
            act3                <= 8'sd0;
            seed0               <= 24'sd0;
            seed1               <= 24'sd0;
            done_latched        <= 1'b0;
            weight_done_latched <= 1'b0;
        end else begin
            if (cmd_load_weight && controller_weight_ready && !controller_weight_done) begin
                weight_done_latched <= 1'b0;
            end

            if (cmd_start && controller_start_ready) begin
                done_latched <= 1'b0;
            end

            if (controller_weight_done) begin
                weight_done_latched <= 1'b1;
            end

            if (controller_done) begin
                done_latched <= 1'b1;
            end

            if (cmd_load_act) begin
                case (index)
                    2'd0: act0 <= uio_in;
                    2'd1: act1 <= uio_in;
                    2'd2: act2 <= uio_in;
                    default: act3 <= uio_in;
                endcase
            end

            if (cmd_load_seed) begin
                if (!row_select) begin
                    case (index)
                        2'd0: seed0[7:0]   <= uio_in;
                        2'd1: seed0[15:8]  <= uio_in;
                        2'd2: seed0[23:16] <= uio_in;
                        default: seed0     <= seed0;
                    endcase
                end else begin
                    case (index)
                        2'd0: seed1[7:0]   <= uio_in;
                        2'd1: seed1[15:8]  <= uio_in;
                        2'd2: seed1[23:16] <= uio_in;
                        default: seed1     <= seed1;
                    endcase
                end
            end
        end
    end

    wire signed [23:0] result0 = controller_result[23:0];
    wire signed [23:0] result1 = controller_result[47:24];
    wire signed [23:0] selected_result = row_select ? result1 : result0;

    wire [7:0] result_byte0 = selected_result[7:0];
    wire [7:0] result_byte1 = selected_result[15:8];
    wire [7:0] result_byte2 = selected_result[23:16];
    wire [7:0] result_byte3 = {8{selected_result[23]}};
    wire [7:0] read_result_byte =
        (index == 2'd0) ? result_byte0 :
        (index == 2'd1) ? result_byte1 :
        (index == 2'd2) ? result_byte2 :
                          result_byte3;

    wire [7:0] status_byte = {
        controller_result_valid[1],
        controller_result_valid[0],
        controller_weight_ready,
        controller_start_ready,
        &controller_result_valid,
        weight_done_latched,
        done_latched,
        controller_busy
    };

    assign uo_out  = cmd_read_result ? read_result_byte : status_byte;
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    w1a8_tile_controller #(
        .ACT_WIDTH (8),
        .PSUM_WIDTH(24),
        .ROWS      (2),
        .COLS      (4)
    ) u_controller (
        .clk              (clk),
        .rst_n            (rst_n),
        .clear            (cmd_clear),
        .weight_load_valid(cmd_load_weight),
        .weight_load_bits (uio_in[1:0]),
        .weight_load_ready(controller_weight_ready),
        .weight_load_done (controller_weight_done),
        .weight_out       (controller_weight_out),
        .start            (cmd_start),
        .start_ready      (controller_start_ready),
        .busy             (controller_busy),
        .done             (controller_done),
        .act_vector       (act_vector),
        .seed_in          (seed_in),
        .result_out       (controller_result),
        .result_valid     (controller_result_valid)
    );

    wire _unused = &{ena, ui_in[7:6], cmd_status, controller_weight_out, 1'b0};

endmodule
