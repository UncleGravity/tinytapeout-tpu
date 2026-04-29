/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * tpu_tile_ctrl - tile FSM and step counter.
 *
 * Sequences three phases per START:
 *   IDLE -> LOAD: shift weights into the array for COLS cycles.
 *   LOAD -> RUN : drive activation skew + per-row valid skew until the
 *                 wavefront has exited every row (step >= ROWS+COLS-1) AND
 *                 acc_mem reports all rows captured.
 *   RUN  -> IDLE: latch DONE; host can RDP and SEED again.
 *
 * Owns timing only. Memory layouts and skew are implemented inside the
 * scratchpad modules (each consumes step + run_phase / weight_load).
 *
 * start_pulse: combinational `idle && cmd_start`, fires on the same edge
 * that state transitions IDLE -> LOAD, so acc_mem can clear acc_done_q in
 * lockstep with the FSM.
 */

`default_nettype none

module tpu_tile_ctrl #(
    parameter ROWS = 2,
    parameter COLS = 2
) (
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              clear,

    input  wire                              cmd_start,
    input  wire                              all_rows_done,

    output wire                              idle,
    output wire                              busy,
    output wire                              load_phase,
    output wire                              run_phase,
    output wire [$clog2(ROWS+COLS)-1:0]      step,

    output wire                              start_pulse,
    output wire                              done_latched,
    output wire                              weight_done_latched
);

    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_LOAD = 2'd1;
    localparam [1:0] STATE_RUN  = 2'd2;

    localparam COMPUTE_LAST_STEP = ROWS + COLS - 1;
    localparam STEP_W            = $clog2(ROWS + COLS);

    logic [1:0]        state_q;
    logic [STEP_W-1:0] step_q;
    logic              done_latched_q;
    logic              weight_done_q;

    assign idle                = (state_q == STATE_IDLE);
    assign busy                = !idle;
    assign load_phase          = (state_q == STATE_LOAD);
    assign run_phase           = (state_q == STATE_RUN);

    assign step                = step_q;
    assign start_pulse         = idle && cmd_start;
    assign done_latched        = done_latched_q;
    assign weight_done_latched = weight_done_q;

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            state_q        <= STATE_IDLE;
            step_q         <= {STEP_W{1'b0}};
            done_latched_q <= 1'b0;
            weight_done_q  <= 1'b0;
        end else begin
            case (state_q)
                STATE_IDLE: begin
                    if (cmd_start) begin
                        state_q        <= STATE_LOAD;
                        step_q         <= {STEP_W{1'b0}};
                        done_latched_q <= 1'b0;
                        weight_done_q  <= 1'b0;
                    end
                end

                STATE_LOAD: begin
                    if (step_q == (COLS - 1)) begin
                        state_q       <= STATE_RUN;
                        step_q        <= {STEP_W{1'b0}};
                        weight_done_q <= 1'b1;
                    end else begin
                        step_q <= step_q + 1'b1;
                    end
                end

                STATE_RUN: begin
                    if (step_q < COMPUTE_LAST_STEP[STEP_W-1:0]) begin
                        step_q <= step_q + 1'b1;
                    end
                    if ((step_q >= COMPUTE_LAST_STEP[STEP_W-1:0]) && all_rows_done) begin
                        state_q        <= STATE_IDLE;
                        done_latched_q <= 1'b1;
                    end
                end

                default: state_q <= STATE_IDLE;
            endcase
        end
    end

endmodule
