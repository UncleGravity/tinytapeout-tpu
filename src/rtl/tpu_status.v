/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_status - combinational uo_out mux + status byte assembly.
 *
 * Status byte layout (LSB first):
 *   [0] busy
 *   [1] done_latched
 *   [2] weight_done_latched
 *   [3] all rows captured (acc_done == all-ones)
 *   [4] start_ready (== idle)
 *   [5] reserved (held high while idle for protocol stability)
 *   [6] error_latched
 *   [7] reserved (low)
 *
 * uo_out drives the result byte during RDP, otherwise the status byte.
 */

`default_nettype none

module tpu_status (
    input  wire       cmd_rdp,
    input  wire [7:0] result_byte,

    input  wire       busy,
    input  wire       idle,
    input  wire       done_latched,
    input  wire       weight_done_latched,
    input  wire       all_rows_done,
    input  wire       error_latched,

    output wire [7:0] uo_out
);

    wire [7:0] status_byte;
    assign status_byte[0] = busy;
    assign status_byte[1] = done_latched;
    assign status_byte[2] = weight_done_latched;
    assign status_byte[3] = all_rows_done;
    assign status_byte[4] = idle;
    assign status_byte[5] = idle;
    assign status_byte[6] = error_latched;
    assign status_byte[7] = 1'b0;

    assign uo_out = cmd_rdp ? result_byte : status_byte;

endmodule
