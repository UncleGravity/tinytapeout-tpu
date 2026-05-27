// q1a8_kernel - rowblock stream parser plus row-parallel Q1A8 datapath.
//
// Stream format, little-endian 64-bit beats, for each Q1 block:
//
//   weight scales:
//     ceil(ROWS/4) beats, four fp16 scales per beat
//
//   repeated for q8_local = 0, 32, 64, 96:
//     4 beats   activation int8[32]
//     1 beat    act_scale in bits [15:0]
//     ceil(ROWS/2) beats, two uint32 weight-bit words per beat
//
// The same activation sub-block is broadcast across ROWS lanes, avoiding the
// old per-cell activation repetition.

`default_nettype none

module q1a8_kernel #(
    parameter integer ROWS = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [7:0]            row_count,
    output reg                   kernel_done,
    output wire                  busy,
    output wire [ROWS*32-1:0]    results_flat,

    input  wire [63:0]           s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready
);

    localparam integer SCALE_BEATS = (ROWS + 3) / 4;
    localparam integer WBITS_BEATS = (ROWS + 1) / 2;

    localparam [2:0] ST_IDLE      = 3'd0;
    localparam [2:0] ST_SCALES    = 3'd1;
    localparam [2:0] ST_ACTS      = 3'd2;
    localparam [2:0] ST_ACT_SCALE = 3'd3;
    localparam [2:0] ST_WBITS     = 3'd4;
    localparam [2:0] ST_ISSUE     = 3'd5;
    localparam [2:0] ST_WAIT_DONE = 3'd6;

    reg [2:0] state;
    reg busy_q;
    reg [15:0] q1_remaining;
    reg [1:0] q8_index;
    reg [7:0] beat_index;

    reg [ROWS*16-1:0] weight_scales_flat;
    reg [ROWS*32-1:0] weight_bits_flat;
    reg [255:0] acts_packed;
    reg [15:0] act_scale;

    reg rowblock_start;
    reg rowblock_valid;
    reg rowblock_last;
    wire rowblock_done;

    assign busy = busy_q;
    assign s_axis_tready =
        busy_q &&
        ((state == ST_SCALES) ||
         (state == ST_ACTS) ||
         (state == ST_ACT_SCALE) ||
         (state == ST_WBITS));

    wire beat_accept = s_axis_tvalid && s_axis_tready;
    wire start_pulse = start_kernel && !busy_q;

    q1a8_rowblock #(.ROWS(ROWS)) u_rowblock (
        .clk(clk),
        .rst_n(rst_n),
        .start(rowblock_start),
        .row_count(row_count),
        .valid_in(rowblock_valid),
        .last_in(rowblock_last),
        .weight_bits_flat(weight_bits_flat),
        .weight_scales_flat(weight_scales_flat),
        .acts_packed(acts_packed),
        .act_scale(act_scale),
        .done(rowblock_done),
        .results_flat(results_flat)
    );

    integer lane;
    always @(posedge clk) begin
        if (!rst_n) begin
            state               <= ST_IDLE;
            busy_q              <= 1'b0;
            q1_remaining        <= 16'd0;
            q8_index            <= 2'd0;
            beat_index          <= 8'd0;
            weight_scales_flat  <= {ROWS*16{1'b0}};
            weight_bits_flat    <= {ROWS*32{1'b0}};
            acts_packed         <= 256'd0;
            act_scale           <= 16'd0;
            rowblock_start      <= 1'b0;
            rowblock_valid      <= 1'b0;
            rowblock_last       <= 1'b0;
            kernel_done         <= 1'b0;
        end else begin
            rowblock_start <= 1'b0;
            rowblock_valid <= 1'b0;
            rowblock_last  <= 1'b0;
            kernel_done    <= 1'b0;

            if (start_pulse) begin
                busy_q         <= 1'b1;
                q1_remaining   <= num_q1_blocks;
                q8_index       <= 2'd0;
                beat_index     <= 8'd0;
                rowblock_start <= 1'b1;
                state          <= ST_SCALES;
            end else if (busy_q) begin
                case (state)
                    ST_SCALES: begin
                        if (beat_accept) begin
                            for (lane = 0; lane < 4; lane = lane + 1) begin
                                if ((beat_index * 4 + lane) < ROWS) begin
                                    weight_scales_flat[(beat_index * 4 + lane) * 16 +: 16]
                                        <= s_axis_tdata[lane * 16 +: 16];
                                end
                            end

                            if ({24'd0, beat_index} == SCALE_BEATS - 1) begin
                                beat_index <= 8'd0;
                                q8_index   <= 2'd0;
                                state      <= ST_ACTS;
                            end else begin
                                beat_index <= beat_index + 8'd1;
                            end
                        end
                    end

                    ST_ACTS: begin
                        if (beat_accept) begin
                            acts_packed[beat_index * 64 +: 64] <= s_axis_tdata;
                            if (beat_index == 8'd3) begin
                                beat_index <= 8'd0;
                                state      <= ST_ACT_SCALE;
                            end else begin
                                beat_index <= beat_index + 8'd1;
                            end
                        end
                    end

                    ST_ACT_SCALE: begin
                        if (beat_accept) begin
                            act_scale  <= s_axis_tdata[15:0];
                            beat_index <= 8'd0;
                            state      <= ST_WBITS;
                        end
                    end

                    ST_WBITS: begin
                        if (beat_accept) begin
                            for (lane = 0; lane < 2; lane = lane + 1) begin
                                if ((beat_index * 2 + lane) < ROWS) begin
                                    weight_bits_flat[(beat_index * 2 + lane) * 32 +: 32]
                                        <= s_axis_tdata[lane * 32 +: 32];
                                end
                            end

                            if ({24'd0, beat_index} == WBITS_BEATS - 1) begin
                                beat_index <= 8'd0;
                                state      <= ST_ISSUE;
                            end else begin
                                beat_index <= beat_index + 8'd1;
                            end
                        end
                    end

                    ST_ISSUE: begin
                        rowblock_valid <= 1'b1;
                        rowblock_last  <= (q1_remaining == 16'd1) && (q8_index == 2'd3);

                        if (q8_index == 2'd3) begin
                            q8_index <= 2'd0;
                            if (q1_remaining == 16'd1) begin
                                state <= ST_WAIT_DONE;
                            end else begin
                                q1_remaining <= q1_remaining - 16'd1;
                                state        <= ST_SCALES;
                            end
                        end else begin
                            q8_index <= q8_index + 2'd1;
                            state    <= ST_ACTS;
                        end
                    end

                    ST_WAIT_DONE: begin
                        if (rowblock_done) begin
                            kernel_done <= 1'b1;
                            busy_q      <= 1'b0;
                            state       <= ST_IDLE;
                        end
                    end

                    default: begin
                        state <= ST_IDLE;
                        busy_q <= 1'b0;
                    end
                endcase
            end
        end
    end

endmodule
