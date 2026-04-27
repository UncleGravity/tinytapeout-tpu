/*
 * Copyright (c) 2026 UncleGravity
 * SPDX-License-Identifier: Apache-2.0
 *
 * w1a8_tile_controller - transaction wrapper for a W1A8 systolic array tile.
 *
 * This module keeps the raw array regular while exposing a simpler compute
 * interface:
 *
 *   1. Load weights through the physical row shift chains.
 *      Present one bit per row per cycle, last column first.
 *
 *   2. Start one COLS-wide activation vector.
 *      The controller generates the column/row skew required by the array.
 *
 *   3. Wait for done.
 *      Raw row outputs are staggered by the array; the controller captures them
 *      into result_out and pulses done once all rows are available.
 */

`default_nettype none

module w1a8_tile_controller #(
    parameter ACT_WIDTH  = 8,
    parameter PSUM_WIDTH = 16,
    parameter ROWS       = 2,
    parameter COLS       = 4
) (
    input  wire                                clk,
    input  wire                                rst_n,
    input  wire                                clear,

    input  wire                                weight_load_valid,
    input  wire [ROWS-1:0]                     weight_load_bits,
    output wire                                weight_load_ready,
    output logic                               weight_load_done,

    input  wire                                start,
    output wire                                start_ready,
    output wire                                busy,
    output logic                               done,

    input  wire signed [COLS*ACT_WIDTH-1:0]    act_vector,
    input  wire signed [ROWS*PSUM_WIDTH-1:0]   seed_in,
    output logic signed [ROWS*PSUM_WIDTH-1:0] result_out,
    output logic [ROWS-1:0]                   result_valid
);

    localparam integer DRIVE_CYCLES = ROWS + COLS - 1;
    localparam integer STEP_WIDTH   = (DRIVE_CYCLES <= 1) ? 1 : $clog2(DRIVE_CYCLES + 1);
    localparam integer LOAD_WIDTH   = (COLS <= 1) ? 1 : $clog2(COLS);

    logic compute_active;
    logic [STEP_WIDTH-1:0] step_q;
    logic [LOAD_WIDTH-1:0] load_count_q;

    logic signed [COLS*ACT_WIDTH-1:0] act_vector_q;
    logic signed [ROWS*PSUM_WIDTH-1:0] seed_q;

    wire  signed [COLS*ACT_WIDTH-1:0] array_act_in;
    wire  signed [COLS*ACT_WIDTH-1:0] array_act_out;
    wire  signed [ROWS*PSUM_WIDTH-1:0] array_psum_in;
    wire  signed [ROWS*PSUM_WIDTH-1:0] array_psum_out;
    wire  [ROWS-1:0] array_valid_in;
    wire  [ROWS-1:0] array_valid_out;

    wire load_accept  = weight_load_valid && weight_load_ready;
    wire start_accept = start && start_ready;
    wire drive_active = compute_active && (step_q < DRIVE_CYCLES[STEP_WIDTH-1:0]);
    wire [ROWS-1:0] collected_valid = result_valid | array_valid_out;

    assign busy              = compute_active;
    assign weight_load_ready = !compute_active;
    assign start_ready       = !compute_active && !weight_load_valid;

    integer capture_row;

    genvar skew_col;
    genvar skew_row;
    generate
        for (skew_col = 0; skew_col < COLS; skew_col = skew_col + 1) begin : gen_act_skew
            assign array_act_in[skew_col*ACT_WIDTH +: ACT_WIDTH] =
                (drive_active && (step_q == skew_col)) ?
                    act_vector_q[skew_col*ACT_WIDTH +: ACT_WIDTH] :
                    {ACT_WIDTH{1'b0}};
        end

        for (skew_row = 0; skew_row < ROWS; skew_row = skew_row + 1) begin : gen_seed_skew
            assign array_valid_in[skew_row] = drive_active && (step_q == skew_row);
            assign array_psum_in[skew_row*PSUM_WIDTH +: PSUM_WIDTH] =
                (drive_active && (step_q == skew_row)) ?
                    seed_q[skew_row*PSUM_WIDTH +: PSUM_WIDTH] :
                    {PSUM_WIDTH{1'b0}};
        end
    endgenerate

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            compute_active   <= 1'b0;
            step_q           <= {STEP_WIDTH{1'b0}};
            load_count_q     <= {LOAD_WIDTH{1'b0}};
            weight_load_done <= 1'b0;
            done             <= 1'b0;
            result_valid     <= {ROWS{1'b0}};
            result_out       <= {ROWS*PSUM_WIDTH{1'b0}};
            act_vector_q     <= {COLS*ACT_WIDTH{1'b0}};
            seed_q           <= {ROWS*PSUM_WIDTH{1'b0}};
        end else if (clear) begin
            compute_active   <= 1'b0;
            step_q           <= {STEP_WIDTH{1'b0}};
            load_count_q     <= {LOAD_WIDTH{1'b0}};
            weight_load_done <= 1'b0;
            done             <= 1'b0;
            result_valid     <= {ROWS{1'b0}};
            result_out       <= {ROWS*PSUM_WIDTH{1'b0}};
            act_vector_q     <= {COLS*ACT_WIDTH{1'b0}};
            seed_q           <= {ROWS*PSUM_WIDTH{1'b0}};
        end else begin
            weight_load_done <= 1'b0;
            done             <= 1'b0;

            if (load_accept) begin
                if (load_count_q == COLS - 1) begin
                    load_count_q     <= {LOAD_WIDTH{1'b0}};
                    weight_load_done <= 1'b1;
                end else begin
                    load_count_q <= load_count_q + 1'b1;
                end
            end

            for (capture_row = 0; capture_row < ROWS; capture_row = capture_row + 1) begin
                if (array_valid_out[capture_row]) begin
                    result_out[capture_row*PSUM_WIDTH +: PSUM_WIDTH] <=
                        array_psum_out[capture_row*PSUM_WIDTH +: PSUM_WIDTH];
                end
            end

            if (start_accept) begin
                compute_active <= 1'b1;
                step_q         <= {STEP_WIDTH{1'b0}};
                result_valid   <= {ROWS{1'b0}};
                result_out     <= {ROWS*PSUM_WIDTH{1'b0}};
                act_vector_q   <= act_vector;
                seed_q         <= seed_in;
            end else if (compute_active) begin
                result_valid <= collected_valid;

                if (step_q < DRIVE_CYCLES[STEP_WIDTH-1:0]) begin
                    step_q <= step_q + 1'b1;
                end

                if ((step_q >= DRIVE_CYCLES[STEP_WIDTH-1:0]) &&
                    (collected_valid == {ROWS{1'b1}})) begin
                    compute_active <= 1'b0;
                    done           <= 1'b1;
                end
            end
        end
    end

    w1a8_array #(
        .ACT_WIDTH (ACT_WIDTH),
        .PSUM_WIDTH(PSUM_WIDTH),
        .ROWS      (ROWS),
        .COLS      (COLS)
    ) u_array (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear      (clear),
        .weight_load(load_accept),
        .weight_in  (weight_load_bits),
        .act_in     (array_act_in),
        .act_out    (array_act_out),
        .psum_in    (array_psum_in),
        .valid_in   (array_valid_in),
        .psum_out   (array_psum_out),
        .valid_out  (array_valid_out)
    );

    wire _unused = &{array_act_out, 1'b0};

endmodule
