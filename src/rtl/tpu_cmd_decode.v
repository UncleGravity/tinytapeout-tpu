/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_cmd_decode - combinational ui_in decoder.
 *
 * Slices the 8-bit command pin field into typed command strobes and a 5-bit
 * argument bus. STATUS (0) and NOP (7) have no effect on the pipeline; their
 * "uo_out reads status" behavior is the default of the output mux, so they
 * don't need a strobe here.
 */

`default_nettype none

module tpu_cmd_decode (
    input  wire [7:0] ui_in,

    output wire       cmd_clear,
    output wire       cmd_ldw,
    output wire       cmd_lda,
    output wire       cmd_seed,
    output wire       cmd_start,
    output wire       cmd_rdp,

    output wire [4:0] arg
);

    localparam [2:0] CMD_CLEAR = 3'd1;
    localparam [2:0] CMD_LDW   = 3'd2;
    localparam [2:0] CMD_LDA   = 3'd3;
    localparam [2:0] CMD_SEED  = 3'd4;
    localparam [2:0] CMD_START = 3'd5;
    localparam [2:0] CMD_RDP   = 3'd6;

    wire [2:0] cmd = ui_in[2:0];

    assign cmd_clear = (cmd == CMD_CLEAR);
    assign cmd_ldw   = (cmd == CMD_LDW);
    assign cmd_lda   = (cmd == CMD_LDA);
    assign cmd_seed  = (cmd == CMD_SEED);
    assign cmd_start = (cmd == CMD_START);
    assign cmd_rdp   = (cmd == CMD_RDP);

    assign arg = ui_in[7:3];

endmodule
