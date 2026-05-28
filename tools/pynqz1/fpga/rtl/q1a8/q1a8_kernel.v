// q1a8_kernel - dual-stream rowblock matmul sequencer (v4).
//
// One host command drives `num_rowblocks * num_q1_blocks * 4` Q8 sub-blocks of
// MACs through the row-parallel datapath. After each rowblock's last
// contribution settles into the accumulator, the kernel emits 8 fp32 results
// as 4 beats of 64-bit AXI-Stream on the master interface (lane-major, 2 fp32
// per beat). TLAST asserts on the final beat of the final rowblock.
//
// Dual-stream input format (v4): acts are sent ONCE per matmul column via
// S_AXIS_ACTS, then weights stream rowblock by rowblock via S_AXIS. The
// kernel loads acts+scales into a small BRAM in the LOAD_ACTS phase, then
// broadcasts them as it walks rowblocks against weight bits.
//
// Acts stream layout, per Q1 block (num_q1_blocks total):
//   4 sub-blocks, each = 4 beats int8 acts (32 B) + 1 beat fp16 act_scale
//   (8 B, scale in low 16 bits). 20 beats per Q1 block.
//
// Weights stream layout, per Q1 block per rowblock:
//   weight_scales: ceil(ROWS/4) beats, four fp16 scales per beat
//   for each of 4 q8 sub-blocks: ceil(ROWS/2) beats of two u32 weight-bit
//   words per beat. 18 beats per Q1 block per rowblock (for ROWS=8).
//
// All ROWS lanes are always active; the host packer is responsible for
// zero-padding inactive lanes in the final rowblock when M % ROWS != 0.

`default_nettype none

module q1a8_kernel #(
    parameter integer ROWS           = 8,
    parameter integer MAX_SUB_INDEX  = 256
) (
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  start_kernel,
    input  wire [15:0]           num_q1_blocks,
    input  wire [15:0]           num_rowblocks,
    output reg                   kernel_done,
    output wire                  busy,

    // AXIS slave: weights stream (weight_scales + wbits).
    input  wire [63:0]           s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    // AXIS slave: acts stream (acts + act_scales), once per column.
    input  wire [63:0]           s_axis_acts_tdata,
    input  wire                  s_axis_acts_tvalid,
    output wire                  s_axis_acts_tready,

    // AXIS master: 8 fp32 results per rowblock, packed 2 fp32 per 64-bit beat.
    output wire [63:0]           m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,
    output wire                  m_axis_tlast,
    output wire [7:0]            m_axis_tkeep
);

    localparam integer SCALE_BEATS    = (ROWS + 3) / 4;
    localparam integer WBITS_BEATS    = (ROWS + 1) / 2;
    localparam integer ACT_BEATS      = 4;          // 32-byte sub-block = 4 × 8 B
    localparam integer Q8_SUBBLOCKS   = 4;          // Q1_BLOCK / Q8_BLOCK

    // FSM. LOAD_ACTS runs once per kernel start; then SCALES/WBITS/ISSUE
    // iterate over rowblocks × q1_blocks × sub-blocks against acts in BRAM.
    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_LOAD_ACTS  = 4'd1;
    localparam [3:0] ST_LOAD_SCALE = 4'd2;
    localparam [3:0] ST_SCALES     = 4'd3;
    localparam [3:0] ST_WBITS      = 4'd4;
    localparam [3:0] ST_ISSUE      = 4'd5;
    localparam [3:0] ST_WAIT_DONE  = 4'd6;
    localparam [3:0] ST_DRAIN      = 4'd7;
    localparam [3:0] ST_EMIT       = 4'd8;
    localparam [3:0] ST_FINISH     = 4'd9;

    reg [3:0]  state;
    reg        busy_q;
    reg [15:0] q1_remaining;
    reg [15:0] rowblock_remaining;
    reg [1:0]  q8_index;             // sub-block within current Q1 block
    reg [7:0]  beat_index;
    reg [1:0]  drain_cnt;
    reg [1:0]  emit_beat;

    // Q1 block index of the rowblock-in-progress. Counts up 0..num_q1_blocks-1
    // so we can address the acts BRAM with (q1_processed_q*4 + q8_index).
    reg [13:0] q1_processed_q;

    // -- Acts BRAM (one matmul column's worth, broadcast across rowblocks) --
    //
    // 256 bits per sub-block (4 acts beats packed lane-major) + a 16-bit
    // fp16 scale per sub-block. Stored under (q1_idx * 4 + sub_idx).
    reg [255:0] acts_mem        [0:MAX_SUB_INDEX-1];
    reg [15:0]  act_scale_mem   [0:MAX_SUB_INDEX-1];

    // Read port: addressed combinationally from (q1_processed_q, q8_index).
    // Q8_SUBBLOCKS == 4 so the multiply is a left-shift by 2; we encode it
    // as concat to keep Verilog widths clean. BRAM output is registered into
    // acts_q / act_scale_q one cycle later so reading in ST_WBITS lets us
    // use the value in ST_ISSUE.
    localparam integer ADDR_W   = $clog2(MAX_SUB_INDEX);
    localparam integer Q1_IDX_W = $clog2(MAX_SUB_INDEX/4);
    wire [ADDR_W-1:0] bram_read_addr =
        {q1_processed_q[Q1_IDX_W-1:0], q8_index};
    reg [255:0] acts_q;
    reg [15:0]  act_scale_q;
    always @(posedge clk) begin
        acts_q       <= acts_mem[bram_read_addr];
        act_scale_q  <= act_scale_mem[bram_read_addr];
    end

    // -- Acts loading scratch -----------------------------------------------
    reg [255:0] acts_load_accum;
    reg [2:0]   acts_load_beat;      // 0..4 (4 acts beats + 1 scale beat)
    reg [13:0]  acts_load_q1;        // Q1 block index being loaded
    reg [1:0]   acts_load_sub;       // sub-block within Q1 block
    wire [ADDR_W-1:0] acts_load_addr =
        {acts_load_q1[Q1_IDX_W-1:0], acts_load_sub};

    // -- Weights stream regs -----------------------------------------------
    reg [ROWS*16-1:0] weight_scales_flat;
    reg [ROWS*32-1:0] weight_bits_flat;

    reg rowblock_start;
    reg rowblock_valid;
    reg rowblock_last;
    wire rowblock_done;
    wire [ROWS*32-1:0] rowblock_results;

    assign busy = busy_q;
    assign s_axis_tready =
        busy_q &&
        ((state == ST_SCALES) || (state == ST_WBITS));
    assign s_axis_acts_tready =
        busy_q && (state == ST_LOAD_ACTS || state == ST_LOAD_SCALE);

    wire weights_beat_accept = s_axis_tvalid && s_axis_tready;
    wire acts_beat_accept    = s_axis_acts_tvalid && s_axis_acts_tready;
    wire start_pulse         = start_kernel && !busy_q;

    q1a8_rowblock #(.ROWS(ROWS)) u_rowblock (
        .clk(clk),
        .rst_n(rst_n),
        .start(rowblock_start),
        .valid_in(rowblock_valid),
        .last_in(rowblock_last),
        .weight_bits_flat(weight_bits_flat),
        .weight_scales_flat(weight_scales_flat),
        .acts_packed(acts_q),
        .act_scale(act_scale_q),
        .done(rowblock_done),
        .results_flat(rowblock_results)
    );

    // -- Result emit mux ---------------------------------------------------
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
            state               <= ST_IDLE;
            busy_q              <= 1'b0;
            q1_remaining        <= 16'd0;
            rowblock_remaining  <= 16'd0;
            q8_index            <= 2'd0;
            beat_index          <= 8'd0;
            drain_cnt           <= 2'd0;
            emit_beat           <= 2'd0;
            q1_processed_q      <= 14'd0;
            weight_scales_flat  <= {ROWS*16{1'b0}};
            weight_bits_flat    <= {ROWS*32{1'b0}};
            rowblock_start      <= 1'b0;
            rowblock_valid      <= 1'b0;
            rowblock_last       <= 1'b0;
            kernel_done         <= 1'b0;
            acts_load_accum     <= 256'd0;
            acts_load_beat      <= 3'd0;
            acts_load_q1        <= 14'd0;
            acts_load_sub       <= 2'd0;
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
                q1_processed_q     <= 14'd0;
                acts_load_beat     <= 3'd0;
                acts_load_q1       <= 14'd0;
                acts_load_sub      <= 2'd0;
                state              <= ST_LOAD_ACTS;
            end else if (busy_q) begin
                case (state)

                    // -- LOAD_ACTS: consume one column of acts into BRAM --
                    // For each sub-block: 4 beats acts (assemble into 256-bit
                    // word) then 1 beat fp16 scale. Total beats per column
                    // = num_q1_blocks * Q8_SUBBLOCKS * 5.
                    ST_LOAD_ACTS: begin
                        if (acts_beat_accept) begin
                            case (acts_load_beat)
                                3'd0: acts_load_accum[ 63:  0] <= s_axis_acts_tdata;
                                3'd1: acts_load_accum[127: 64] <= s_axis_acts_tdata;
                                3'd2: acts_load_accum[191:128] <= s_axis_acts_tdata;
                                3'd3: acts_load_accum[255:192] <= s_axis_acts_tdata;
                                default: ; // unreachable; scale handled in ST_LOAD_SCALE
                            endcase
                            if (acts_load_beat == 3'd3) begin
                                // Last acts beat — commit accumulator to BRAM
                                // using the just-assembled value via blocking
                                // assignment isn't an option; use a 1-cycle
                                // shim state ST_LOAD_SCALE to settle and
                                // consume the scale beat.
                                acts_load_beat <= 3'd4;
                                state          <= ST_LOAD_SCALE;
                            end else begin
                                acts_load_beat <= acts_load_beat + 3'd1;
                            end
                        end
                    end

                    // Consume one scale beat. acts_load_accum has the full
                    // 256-bit acts value from ST_LOAD_ACTS; write both to
                    // BRAM now. Advance to next sub-block or finish loading.
                    ST_LOAD_SCALE: begin
                        if (acts_beat_accept) begin
                            acts_mem[acts_load_addr]      <= acts_load_accum;
                            act_scale_mem[acts_load_addr] <= s_axis_acts_tdata[15:0];

                            acts_load_beat <= 3'd0;
                            if (acts_load_sub == 2'd3) begin
                                acts_load_sub <= 2'd0;
                                if ({2'b00, acts_load_q1} + 16'd1 == num_q1_blocks) begin
                                    // All sub-blocks loaded — start matmul.
                                    rowblock_start <= 1'b1;
                                    state          <= ST_SCALES;
                                end else begin
                                    acts_load_q1 <= acts_load_q1 + 14'd1;
                                    state        <= ST_LOAD_ACTS;
                                end
                            end else begin
                                acts_load_sub <= acts_load_sub + 2'd1;
                                state         <= ST_LOAD_ACTS;
                            end
                        end
                    end

                    // -- Weight stream consumption -----------------------
                    ST_SCALES: begin
                        if (weights_beat_accept) begin
                            for (lane = 0; lane < 4; lane = lane + 1) begin
                                if ((beat_index * 4 + lane) < ROWS) begin
                                    weight_scales_flat[(beat_index * 4 + lane) * 16 +: 16]
                                        <= s_axis_tdata[lane * 16 +: 16];
                                end
                            end

                            if ({24'd0, beat_index} == SCALE_BEATS - 1) begin
                                beat_index <= 8'd0;
                                q8_index   <= 2'd0;
                                state      <= ST_WBITS;
                            end else begin
                                beat_index <= beat_index + 8'd1;
                            end
                        end
                    end

                    ST_WBITS: begin
                        if (weights_beat_accept) begin
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

                    // -- Compute issue. acts_q / act_scale_q come from BRAM
                    // (registered output of the read launched in ST_WBITS).
                    ST_ISSUE: begin
                        rowblock_valid <= 1'b1;
                        rowblock_last  <= (q1_remaining == 16'd1) && (q8_index == 2'd3);

                        if (q8_index == 2'd3) begin
                            q8_index <= 2'd0;
                            if (q1_remaining == 16'd1) begin
                                state <= ST_WAIT_DONE;
                            end else begin
                                q1_remaining   <= q1_remaining - 16'd1;
                                q1_processed_q <= q1_processed_q + 14'd1;
                                state          <= ST_SCALES;
                            end
                        end else begin
                            q8_index <= q8_index + 2'd1;
                            state    <= ST_WBITS;
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
                                    q1_processed_q     <= 14'd0;
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
