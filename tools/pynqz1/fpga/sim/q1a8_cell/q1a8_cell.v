// One Q1A8 matmul output cell.
//
// Composes q1a8_reducer (one Q8 sub-block per cycle, 1-cycle latency) with
// an fp32 accumulator. Drives one (row, col) cell of the eventual systolic
// array by accumulating K/32 reducer contributions into one fp32 result.
//
// Protocol:
//   1. Pulse `start_cell` for one cycle. acc clears to +0, busy goes high.
//   2. For each of K/32 sub-blocks, raise `valid_in` for one cycle along
//      with the four operand inputs. The reducer + accumulator pipeline
//      absorbs them one cycle apart.
//   3. On the final sub-block's `valid_in` cycle, also raise `last_in`.
//   4. Two cycles later, `cell_done` pulses for one cycle. On that cycle
//      `acc` holds the final fp32 value and `busy` falls.
//
// Re-starting (`start_cell` during busy) is undefined - the accumulator
// will reset but the in-flight reducer output may corrupt the next cell.
// Sequencers should wait for `cell_done` before issuing the next `start_cell`.

`default_nettype none

module q1a8_cell (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         start_cell,
    input  wire         valid_in,
    input  wire         last_in,
    input  wire [31:0]  weight_bits,
    input  wire [255:0] acts_packed,
    input  wire [15:0]  weight_scale,
    input  wire [15:0]  act_scale,

    output wire         cell_done,
    output reg          busy,
    output wire [31:0]  acc
);
    // -- Reducer (validated by test_q1a8_reducer) ------------------------
    wire        reducer_valid;
    wire [31:0] contribution;
    q1a8_reducer u_reducer (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_in),
        .weight_bits(weight_bits),
        .acts_packed(acts_packed),
        .weight_scale(weight_scale),
        .act_scale(act_scale),
        .valid_out(reducer_valid),
        .contribution(contribution)
    );

    // -- fp32 accumulator ------------------------------------------------
    reg  [31:0] acc_q;
    wire [31:0] acc_next;
    fp32_add u_add (.a(acc_q), .b(contribution), .out(acc_next));

    always @(posedge clk) begin
        if (!rst_n)             acc_q <= 32'd0;
        else if (start_cell)    acc_q <= 32'd0;
        else if (reducer_valid) acc_q <= acc_next;
    end

    // -- Track last sub-block through reducer + accumulator pipeline -----
    // last_in pulses on cycle N (with valid_in). On cycle N+1 the reducer
    // emits the last contribution, the accumulator latches the new value
    // on the edge of cycle N+2, and cell_done pulses on cycle N+2 so the
    // host sees `cell_done && acc` together.
    reg last_d1, last_d2;
    always @(posedge clk) begin
        if (!rst_n || start_cell) begin
            last_d1 <= 1'b0;
            last_d2 <= 1'b0;
        end else begin
            last_d1 <= valid_in && last_in;
            last_d2 <= last_d1;
        end
    end

    assign cell_done = last_d2;
    assign acc       = acc_q;

    // -- Busy ------------------------------------------------------------
    // Drop on `last_d1` (one cycle ahead of `cell_done`) so that on the
    // exact cycle `cell_done` pulses, `busy` is already 0. That way a
    // downstream controller can use either signal as the "compute done"
    // edge without an off-by-one ambiguity.
    always @(posedge clk) begin
        if (!rst_n)          busy <= 1'b0;
        else if (start_cell) busy <= 1'b1;
        else if (last_d1)    busy <= 1'b0;
    end
endmodule
