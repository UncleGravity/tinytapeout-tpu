// q1a8_streamer - wraps q1a8_cell with the stream-driven FSM that the
// eventual AXI-DMA-fed PL kernel will use.
//
// Two interfaces:
//
//   Kernel control (looks like AXI-Lite registers):
//     start_kernel    pulse to begin a new accumulation
//     num_subblocks   how many sub-blocks (K/32) for this kernel
//     kernel_done     1-cycle pulse when result is latched
//     busy            high from start_kernel until kernel_done
//     result          fp32 accumulator value, valid when kernel_done=1
//
//   Sub-block input stream (AXIS-like, ready/valid handshake):
//     s_valid / s_ready
//     s_weight_bits, s_acts_packed, s_weight_scale, s_act_scale
//
// The downstream cell already accepts one sub-block per cycle, so s_ready
// is just "we're running and still have sub-blocks to consume". When the
// upstream stalls (s_valid=0), the cell stalls too: nothing is fed, the
// accumulator doesn't advance, no state corruption. When the upstream
// resumes, accumulation continues from where it left off.
//
// On the cycle that the last sub-block is accepted, the FSM raises the
// cell's last_in. Two cycles later (reducer + accumulator pipeline depth)
// the cell pulses cell_done. The streamer latches `result` and pulses
// `kernel_done` on the NEXT cycle, so the host sees them together.

`default_nettype none

module q1a8_streamer (
    input  wire         clk,
    input  wire         rst_n,

    // Kernel control
    input  wire         start_kernel,
    input  wire [15:0]  num_subblocks,
    output reg          kernel_done,
    output wire         busy,
    output reg  [31:0]  result,

    // Sub-block input stream (AXIS-like handshake)
    input  wire         s_valid,
    output wire         s_ready,
    input  wire [31:0]  s_weight_bits,
    input  wire [255:0] s_acts_packed,
    input  wire [15:0]  s_weight_scale,
    input  wire [15:0]  s_act_scale
);
    // -- State -----------------------------------------------------------
    reg         busy_q;
    reg [15:0]  remaining;

    // start_pulse is combinational on start_kernel so the cell gets its
    // start_cell on the same edge that we transition into busy. Spurious
    // start_kernel during busy is ignored (no nested-kernel support).
    wire start_pulse = start_kernel && !busy_q;

    // Handshake.
    wire accepting = s_valid && busy_q && (remaining != 16'd0);
    wire last_sb   = accepting && (remaining == 16'd1);

    assign s_ready = busy_q && (remaining != 16'd0);
    assign busy    = busy_q;

    // -- Cell ------------------------------------------------------------
    wire        cell_done;
    wire [31:0] cell_acc;

    q1a8_cell u_cell (
        .clk(clk), .rst_n(rst_n),
        .start_cell(start_pulse),
        .valid_in(accepting),
        .last_in(last_sb),
        .weight_bits(s_weight_bits),
        .acts_packed(s_acts_packed),
        .weight_scale(s_weight_scale),
        .act_scale(s_act_scale),
        .cell_done(cell_done),
        .busy(/* unused; we drive our own */),
        .acc(cell_acc)
    );

    // -- Control ---------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            busy_q      <= 1'b0;
            remaining   <= 16'd0;
            result      <= 32'd0;
            kernel_done <= 1'b0;
        end else begin
            // kernel_done is a 1-cycle pulse by default.
            kernel_done <= 1'b0;

            if (start_pulse) begin
                busy_q    <= 1'b1;
                remaining <= num_subblocks;
            end

            // Accept and start_pulse can never co-occur (start_pulse requires
            // !busy_q, accepting requires busy_q), so this branch is safe.
            if (accepting) begin
                remaining <= remaining - 16'd1;
            end

            if (cell_done && busy_q) begin
                result      <= cell_acc;
                kernel_done <= 1'b1;
                busy_q      <= 1'b0;
            end
        end
    end
endmodule
