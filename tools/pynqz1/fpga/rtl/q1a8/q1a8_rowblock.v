// q1a8_rowblock - accumulate one activation column against several rows.
//
// This is the replacement for the old one-cell datapath. Each valid input is
// one Q8 sub-block shared by ROWS output rows. Every lane has its own 32 Q1
// bits and fp16 weight scale, while the 32 int8 activations and fp16
// activation scale are broadcast across lanes.
//
// The output order is lane-major:
//   results_flat[31:0]     = lane 0 fp32 bits
//   results_flat[63:32]    = lane 1 fp32 bits
//   ...

`default_nettype none

module q1a8_rowblock #(
    parameter integer ROWS = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start,
    input  wire [7:0]            row_count,

    input  wire                  valid_in,
    input  wire                  last_in,
    input  wire [ROWS*32-1:0]    weight_bits_flat,
    input  wire [ROWS*16-1:0]    weight_scales_flat,
    input  wire [255:0]          acts_packed,
    input  wire [15:0]           act_scale,

    output reg                   done,
    output wire [ROWS*32-1:0]    results_flat
);

    localparam integer REDUCER_LATENCY = 2;

    wire [ROWS-1:0] reducer_valid;
    wire [ROWS*32-1:0] contributions_flat;
    wire [ROWS*32-1:0] add_results_flat;
    reg  [ROWS*32-1:0] acc_flat;
    reg  [REDUCER_LATENCY-1:0] last_pipe;

    genvar row;
    generate
        for (row = 0; row < ROWS; row = row + 1) begin : gen_lanes
            wire lane_active = row_count > row;
            wire [31:0] acc = acc_flat[row*32 +: 32];
            wire [31:0] contribution = contributions_flat[row*32 +: 32];

            q1a8_reducer u_reducer (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_in && lane_active),
                .weight_bits(weight_bits_flat[row*32 +: 32]),
                .acts_packed(acts_packed),
                .weight_scale(weight_scales_flat[row*16 +: 16]),
                .act_scale(act_scale),
                .valid_out(reducer_valid[row]),
                .contribution(contributions_flat[row*32 +: 32])
            );

            fp32_add u_acc_add (
                .a(acc),
                .b(contribution),
                .out(add_results_flat[row*32 +: 32])
            );
        end
    endgenerate

    assign results_flat = acc_flat;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_flat <= {ROWS*32{1'b0}};
            last_pipe <= {REDUCER_LATENCY{1'b0}};
            done     <= 1'b0;
        end else begin
            done      <= 1'b0;
            last_pipe <= {last_pipe[REDUCER_LATENCY-2:0], valid_in && last_in};

            if (start) begin
                acc_flat  <= {ROWS*32{1'b0}};
                last_pipe <= {REDUCER_LATENCY{1'b0}};
            end else begin
                for (i = 0; i < ROWS; i = i + 1) begin
                    if (reducer_valid[i]) begin
                        acc_flat[i*32 +: 32] <= add_results_flat[i*32 +: 32];
                    end
                end

                if (last_pipe[REDUCER_LATENCY-1]) begin
                    done <= 1'b1;
                end
            end
        end
    end

endmodule
