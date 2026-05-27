// q1a8_kernel - rowblock stream parser + multi-rowblock matmul sequencer.
//
// One host command drives `num_rowblocks * num_q1_blocks * 4` Q8 sub-blocks of
// MACs through the row-parallel datapath. After each rowblock's last
// contribution settles into the accumulator, the kernel emits 8 fp32 results
// as 4 beats of 64-bit AXI-Stream on the master interface (lane-major, 2 fp32
// per beat). TLAST asserts on the final beat of the final rowblock.
//
// Stream input format, little-endian 64-bit beats, for each Q1 block:
//   weight scales: ceil(ROWS/4) beats, four fp16 scales per beat
//   for each q8_local in (0, 32, 64, 96):
//     4 beats   activation int8[32]
//     1 beat    act_scale in bits [15:0]
//     ceil(ROWS/2) beats, two uint32 weight-bit words per beat
//
// All ROWS lanes are always active; the host packer is responsible for
// zero-padding inactive lanes in the final rowblock when M % ROWS != 0.

`default_nettype none

module q1a8_kernel #(
    parameter integer ROWS = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [15:0]           num_rowblocks,
    output reg                   kernel_done,
    output wire                  busy,

    // AXIS slave: weight / activation stream.
    input  wire [63:0]           s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    // AXIS master: 8 fp32 results per rowblock, packed 2 fp32 per 64-bit beat.
    output wire [63:0]           m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast,
    output wire [7:0]            m_axis_tkeep
);

    localparam integer SCALE_BEATS = (ROWS + 3) / 4;
    localparam integer WBITS_BEATS = (ROWS + 1) / 2;

    localparam [3:0] ST_IDLE      = 4'd0;
    localparam [3:0] ST_SCALES    = 4'd1;
    localparam [3:0] ST_ACTS      = 4'd2;
    localparam [3:0] ST_ACT_SCALE = 4'd3;
    localparam [3:0] ST_WBITS     = 4'd4;
    localparam [3:0] ST_ISSUE     = 4'd5;
    localparam [3:0] ST_WAIT_DONE = 4'd6;
    localparam [3:0] ST_DRAIN     = 4'd7;
    localparam [3:0] ST_EMIT      = 4'd8;
    localparam [3:0] ST_FINISH    = 4'd9;

    reg [3:0]  state;
    reg        busy_q;
    reg [15:0] q1_remaining;
    reg [15:0] rowblock_remaining;
    reg [1:0]  q8_index;
    reg [7:0]  beat_index;
    reg [1:0]  drain_cnt;
    reg [1:0]  emit_beat;

    reg [ROWS*16-1:0] weight_scales_flat;
    reg [ROWS*32-1:0] weight_bits_flat;
    reg [255:0] acts_packed;
    reg [15:0] act_scale;

    reg rowblock_start;
    reg rowblock_valid;
    reg rowblock_last;
    wire rowblock_done;
    wire [ROWS*32-1:0] rowblock_results;

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
        .valid_in(rowblock_valid),
        .last_in(rowblock_last),
        .weight_bits_flat(weight_bits_flat),
        .weight_scales_flat(weight_scales_flat),
        .acts_packed(acts_packed),
        .act_scale(act_scale),
        .done(rowblock_done),
        .results_flat(rowblock_results)
    );

    // Result emit mux: pack 2 fp32 per 64-bit beat in lane-major order.
    // emit_beat 0 -> {lane1, lane0}, 1 -> {lane3, lane2}, ...
    reg [63:0] emit_word;
    always @(*) begin
        case (emit_beat)
            2'd0:    emit_word = {rowblock_results[ 63: 32], rowblock_results[ 31:  0]};
            2'd1:    emit_word = {rowblock_results[127: 96], rowblock_results[ 95: 64]};
            2'd2:    emit_word = {rowblock_results[191:160], rowblock_results[159:128]};
            default: emit_word = {rowblock_results[255:224], rowblock_results[223:192]};
        endcase
    end

    assign m_axis_tdata  = emit_word;
    assign m_axis_tvalid = (state == ST_EMIT);
    assign m_axis_tlast  = (state == ST_EMIT) &&
                           (emit_beat == 2'd3) &&
                           (rowblock_remaining == 16'd1);
    assign m_axis_tkeep  = 8'hFF;

    integer lane;
    always @(posedge clk) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            busy_q             <= 1'b0;
            q1_remaining       <= 16'd0;
            rowblock_remaining <= 16'd0;
            q8_index           <= 2'd0;
            beat_index         <= 8'd0;
            drain_cnt          <= 2'd0;
            emit_beat          <= 2'd0;
            weight_scales_flat <= {ROWS*16{1'b0}};
            weight_bits_flat   <= {ROWS*32{1'b0}};
            acts_packed        <= 256'd0;
            act_scale          <= 16'd0;
            rowblock_start     <= 1'b0;
            rowblock_valid     <= 1'b0;
            rowblock_last      <= 1'b0;
            kernel_done        <= 1'b0;
        end else begin
            rowblock_start <= 1'b0;
            rowblock_valid <= 1'b0;
            rowblock_last  <= 1'b0;
            kernel_done    <= 1'b0;

            if (start_pulse) begin
                busy_q             <= 1'b1;
                q1_remaining       <= num_q1_blocks;
                rowblock_remaining <= num_rowblocks;
                q8_index           <= 2'd0;
                beat_index         <= 8'd0;
                rowblock_start     <= 1'b1;
                state              <= ST_SCALES;
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
                            // Slack so the last acc_flat update commits before
                            // emit_word reads it.
                            drain_cnt <= 2'd2;
                            state     <= ST_DRAIN;
                        end
                    end

                    ST_DRAIN: begin
                        if (drain_cnt == 2'd0) begin
                            emit_beat <= 2'd0;
                            state     <= ST_EMIT;
                        end else begin
                            drain_cnt <= drain_cnt - 2'd1;
                        end
                    end

                    ST_EMIT: begin
                        if (m_axis_tready) begin
                            if (emit_beat == 2'd3) begin
                                if (rowblock_remaining == 16'd1) begin
                                    state <= ST_FINISH;
                                end else begin
                                    rowblock_remaining <= rowblock_remaining - 16'd1;
                                    q1_remaining       <= num_q1_blocks;
                                    q8_index           <= 2'd0;
                                    beat_index         <= 8'd0;
                                    rowblock_start     <= 1'b1;
                                    state              <= ST_SCALES;
                                end
                            end else begin
                                emit_beat <= emit_beat + 2'd1;
                            end
                        end
                    end

                    ST_FINISH: begin
                        kernel_done <= 1'b1;
                        busy_q      <= 1'b0;
                        state       <= ST_IDLE;
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
