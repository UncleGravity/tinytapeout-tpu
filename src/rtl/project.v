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
 *   1 CLEAR     clear wrapper/controller state
 *   2 SET_ADDR  arg[1:0] selects address register, uio_in is value
 *   3 WRITE     write uio_in to selected bank/address
 *   4 READ      uo_out reads selected bank/address
 *   5 START     load stored weights into the systolic tile and start compute
 *   6 CONFIG    uio_in[2:0] = auto-inc {byte,row,col}
 *   7 NOP       uo_out shows status
 *
 * Address registers:
 *   0 row_addr   output row / weight row
 *   1 col_addr   activation column / first packed weight column
 *   2 byte_addr  seed/result byte
 *   3 bank_addr  selected bank
 *
 * Banks:
 *   0 CONFIG
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
 *   [4] start_ready
 *   [5] weight_load_ready
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

    localparam integer ACT_WIDTH      = 8;
    localparam integer PSUM_WIDTH     = 16;
    localparam integer ROWS           = 2;
    localparam integer COLS           = 4;
    localparam integer ROW_ADDR_WIDTH = 4;
    localparam integer COL_ADDR_WIDTH = 4;
    localparam integer BYTE_ADDR_WIDTH = 3;
    localparam integer BANK_WIDTH     = 3;
    localparam integer LOAD_COL_WIDTH = (COLS <= 1) ? 1 : $clog2(COLS);

    localparam CMD_STATUS   = 3'd0;
    localparam CMD_CLEAR    = 3'd1;
    localparam CMD_SET_ADDR = 3'd2;
    localparam CMD_WRITE    = 3'd3;
    localparam CMD_READ     = 3'd4;
    localparam CMD_START    = 3'd5;
    localparam CMD_CONFIG   = 3'd6;
    localparam CMD_NOP      = 3'd7;

    localparam ADDR_ROW  = 2'd0;
    localparam ADDR_COL  = 2'd1;
    localparam ADDR_BYTE = 2'd2;
    localparam ADDR_BANK = 2'd3;

    localparam BANK_CONFIG = 3'd0;
    localparam BANK_WEIGHT = 3'd1;
    localparam BANK_ACT    = 3'd2;
    localparam BANK_SEED   = 3'd3;
    localparam BANK_RESULT = 3'd4;
    localparam BANK_STATUS = 3'd5;

    localparam STATE_IDLE = 2'd0;
    localparam STATE_LOAD = 2'd1;
    localparam STATE_KICK = 2'd2;
    localparam STATE_RUN  = 2'd3;

    wire [2:0] cmd = ui_in[2:0];
    wire [4:0] arg = ui_in[7:3];

    wire cmd_status   = (cmd == CMD_STATUS);
    wire cmd_clear    = (cmd == CMD_CLEAR);
    wire cmd_set_addr = (cmd == CMD_SET_ADDR);
    wire cmd_write    = (cmd == CMD_WRITE);
    wire cmd_read     = (cmd == CMD_READ);
    wire cmd_start    = (cmd == CMD_START);
    wire cmd_config   = (cmd == CMD_CONFIG);
    wire cmd_nop      = (cmd == CMD_NOP);

    logic [ROW_ADDR_WIDTH-1:0]  row_addr;
    logic [COL_ADDR_WIDTH-1:0]  col_addr;
    logic [BYTE_ADDR_WIDTH-1:0] byte_addr;
    logic [BANK_WIDTH-1:0]      bank_addr;

    logic auto_inc_col;
    logic auto_inc_row;
    logic auto_inc_byte;

    logic [1:0] state_q;
    logic [LOAD_COL_WIDTH-1:0] load_col_q;

    logic weight_mem [0:ROWS-1][0:COLS-1];
    logic signed [ACT_WIDTH-1:0] act_mem [0:COLS-1];
    logic signed [PSUM_WIDTH-1:0] seed_mem [0:ROWS-1];

    wire signed [COLS*ACT_WIDTH-1:0] act_vector;
    wire signed [ROWS*PSUM_WIDTH-1:0] seed_in;

    wire controller_start_ready;
    wire controller_weight_ready;
    wire controller_busy;
    wire controller_done;
    wire [ROWS-1:0] controller_result_valid;
    wire signed [ROWS*PSUM_WIDTH-1:0] controller_result;

    logic done_latched;
    logic weight_done_latched;
    logic error_latched;

    wire busy = (state_q != STATE_IDLE) || controller_busy;
    wire weight_load_drive = (state_q == STATE_LOAD) && controller_weight_ready;
    wire start_drive = (state_q == STATE_KICK) && controller_start_ready;

    logic [ROWS-1:0] weight_load_bits;
    logic [7:0] read_data;

    genvar pack_col;
    generate
        for (pack_col = 0; pack_col < COLS; pack_col = pack_col + 1) begin : gen_pack_act
            assign act_vector[pack_col*ACT_WIDTH +: ACT_WIDTH] = act_mem[pack_col];
        end
    endgenerate

    genvar pack_row;
    generate
        for (pack_row = 0; pack_row < ROWS; pack_row = pack_row + 1) begin : gen_pack_seed
            assign seed_in[pack_row*PSUM_WIDTH +: PSUM_WIDTH] = seed_mem[pack_row];
        end
    endgenerate

    always_comb begin
        weight_load_bits = {ROWS{1'b0}};
        for (int row_i = 0; row_i < ROWS; row_i = row_i + 1) begin
            weight_load_bits[row_i] = weight_mem[row_i][load_col_q];
        end
    end

    wire signed [PSUM_WIDTH-1:0] selected_result =
        (row_addr < ROWS) ?
            controller_result[row_addr*PSUM_WIDTH +: PSUM_WIDTH] :
            {PSUM_WIDTH{1'b0}};
    wire signed [PSUM_WIDTH-1:0] selected_seed =
        (row_addr < ROWS) ?
            seed_mem[row_addr] :
            {PSUM_WIDTH{1'b0}};

    wire [7:0] result_byte0 = selected_result[7:0];
    wire [7:0] result_byte1 = selected_result[15:8];
    wire [7:0] result_sign_byte = {8{selected_result[PSUM_WIDTH-1]}};
    wire [7:0] seed_byte0 = selected_seed[7:0];
    wire [7:0] seed_byte1 = selected_seed[15:8];
    wire [7:0] seed_sign_byte = {8{selected_seed[PSUM_WIDTH-1]}};

    wire [7:0] status_byte = {
        1'b0,
        error_latched,
        controller_weight_ready,
        controller_start_ready && (state_q == STATE_IDLE),
        &controller_result_valid,
        weight_done_latched,
        done_latched,
        busy
    };

    always_comb begin
        read_data = status_byte;
        case (bank_addr)
            BANK_CONFIG: read_data = {5'b0, auto_inc_byte, auto_inc_row, auto_inc_col};
            BANK_ACT: begin
                if (col_addr < COLS) begin
                    read_data = act_mem[col_addr];
                end else begin
                    read_data = 8'h00;
                end
            end
            BANK_SEED: begin
                if (row_addr < ROWS) begin
                    read_data = (byte_addr == 3'd0) ? seed_byte0 :
                                (byte_addr == 3'd1) ? seed_byte1 :
                                                      seed_sign_byte;
                end else begin
                    read_data = 8'h00;
                end
            end
            BANK_RESULT: begin
                read_data = (byte_addr == 3'd0) ? result_byte0 :
                            (byte_addr == 3'd1) ? result_byte1 :
                                                  result_sign_byte;
            end
            BANK_STATUS: read_data = status_byte;
            default: read_data = status_byte;
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            row_addr            <= {ROW_ADDR_WIDTH{1'b0}};
            col_addr            <= {COL_ADDR_WIDTH{1'b0}};
            byte_addr           <= {BYTE_ADDR_WIDTH{1'b0}};
            bank_addr           <= BANK_STATUS;
            auto_inc_col        <= 1'b0;
            auto_inc_row        <= 1'b0;
            auto_inc_byte       <= 1'b0;
            state_q             <= STATE_IDLE;
            load_col_q          <= {LOAD_COL_WIDTH{1'b0}};
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
            load_col_q          <= {LOAD_COL_WIDTH{1'b0}};
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
            if (controller_done) begin
                done_latched <= 1'b1;
            end

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

                    if (cmd_config) begin
                        auto_inc_col  <= uio_in[0];
                        auto_inc_row  <= uio_in[1];
                        auto_inc_byte <= uio_in[2];
                    end

                    if (cmd_write) begin
                        case (bank_addr)
                            BANK_CONFIG: begin
                                auto_inc_col  <= uio_in[0];
                                auto_inc_row  <= uio_in[1];
                                auto_inc_byte <= uio_in[2];
                            end
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
                                    case (byte_addr)
                                        3'd0: seed_mem[row_addr][7:0]  <= uio_in;
                                        3'd1: seed_mem[row_addr][15:8] <= uio_in;
                                        default: seed_mem[row_addr]    <= seed_mem[row_addr];
                                    endcase
                                end else begin
                                    error_latched <= 1'b1;
                                end
                            end
                            default: begin
                            end
                        endcase

                        if (auto_inc_col) begin
                            col_addr <= (bank_addr == BANK_WEIGHT) ? col_addr + 4'd8 : col_addr + 1'b1;
                        end
                        if (auto_inc_row) begin
                            row_addr <= row_addr + 1'b1;
                        end
                        if (auto_inc_byte) begin
                            byte_addr <= byte_addr + 1'b1;
                        end
                    end

                    if (cmd_read) begin
                        if (auto_inc_col) begin
                            col_addr <= col_addr + 1'b1;
                        end
                        if (auto_inc_row) begin
                            row_addr <= row_addr + 1'b1;
                        end
                        if (auto_inc_byte) begin
                            byte_addr <= byte_addr + 1'b1;
                        end
                    end

                    if (cmd_start) begin
                        done_latched        <= 1'b0;
                        weight_done_latched <= 1'b0;
                        state_q             <= STATE_LOAD;
                        load_col_q          <= COLS - 1;
                    end
                end

                STATE_LOAD: begin
                    if (controller_weight_ready) begin
                        if (load_col_q == {LOAD_COL_WIDTH{1'b0}}) begin
                            state_q             <= STATE_KICK;
                            weight_done_latched <= 1'b1;
                        end else begin
                            load_col_q <= load_col_q - 1'b1;
                        end
                    end
                end

                STATE_KICK: begin
                    if (controller_start_ready) begin
                        state_q <= STATE_RUN;
                    end
                end

                STATE_RUN: begin
                    if (controller_done) begin
                        state_q <= STATE_IDLE;
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

    w1a8_tile_controller #(
        .ACT_WIDTH (ACT_WIDTH),
        .PSUM_WIDTH(PSUM_WIDTH),
        .ROWS      (ROWS),
        .COLS      (COLS)
    ) u_controller (
        .clk              (clk),
        .rst_n            (rst_n),
        .clear            (cmd_clear),
        .weight_load_valid(weight_load_drive),
        .weight_load_bits (weight_load_bits),
        .weight_load_ready(controller_weight_ready),
        .weight_load_done (),
        .start            (start_drive),
        .start_ready      (controller_start_ready),
        .busy             (controller_busy),
        .done             (controller_done),
        .act_vector       (act_vector),
        .seed_in          (seed_in),
        .result_out       (controller_result),
        .result_valid     (controller_result_valid)
    );

    wire _unused = &{ena, cmd_status, cmd_nop, 1'b0};

endmodule
