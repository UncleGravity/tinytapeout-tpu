// axis_to_subblock - assemble 64-bit AXIS beats into 320-bit Q8 sub-blocks.
//
// Pack format (little-endian within each beat; one sub-block = 48 bytes = 6 beats):
//
//   beat 0:  {32'b0, weight_bits[31:0]}
//   beat 1:  acts_packed[ 63: 0]
//   beat 2:  acts_packed[127:64]
//   beat 3:  acts_packed[191:128]
//   beat 4:  acts_packed[255:192]
//   beat 5:  {32'b0, act_scale[15:0], weight_scale[15:0]}
//
// The 8-byte alignment makes the host side a flat memcpy from a packed
// uint8[48 * num_subblocks] buffer; no per-beat bookkeeping.
//
// Downstream interface mirrors the streamer's stream input: a 320-bit
// parallel bus with AXIS-style m_valid / m_ready handshake. When the
// 6th beat is accepted, we raise m_valid and hold all the assembled
// fields stable until the downstream consumes them.
//
// One-cycle bubble per sub-block: while m_valid=1 we drop s_axis_tready,
// so the next beat 0 isn't latched until the cycle AFTER consume. At
// 6 beats per sub-block that's ~14% throughput overhead - acceptable
// for the simplicity. A skid buffer would close the gap if needed later.

`default_nettype none

module axis_to_subblock (
    input  wire         clk,
    input  wire         rst_n,

    // AXIS input (64-bit beats, e.g. from an AXI DMA M_AXIS_MM2S).
    input  wire [63:0]  s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,

    // 320-bit parallel sub-block out (AXIS-style handshake).
    output wire         m_valid,
    input  wire         m_ready,
    output wire [31:0]  m_weight_bits,
    output wire [255:0] m_acts_packed,
    output wire [15:0]  m_weight_scale,
    output wire [15:0]  m_act_scale
);
    // -- State -----------------------------------------------------------
    reg [2:0]   phase;             // beat 0..5 expected next
    reg         have_subblock;     // a complete sub-block is held at the output
    reg [31:0]  weight_bits_r;
    reg [255:0] acts_r;
    reg [15:0]  weight_scale_r;
    reg [15:0]  act_scale_r;

    // -- Handshake -------------------------------------------------------
    // We can take a beat only when we're not currently holding one.
    assign s_axis_tready = !have_subblock;
    assign m_valid       = have_subblock;
    assign m_weight_bits = weight_bits_r;
    assign m_acts_packed = acts_r;
    assign m_weight_scale = weight_scale_r;
    assign m_act_scale   = act_scale_r;

    wire beat_accept = s_axis_tvalid && s_axis_tready;
    wire consume     = m_valid && m_ready;

    // -- Control ---------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            phase          <= 3'd0;
            have_subblock  <= 1'b0;
            weight_bits_r  <= 32'd0;
            acts_r         <= 256'd0;
            weight_scale_r <= 16'd0;
            act_scale_r    <= 16'd0;
        end else begin
            // Downstream handshake: drop the held sub-block, reset for the
            // next one. (beat_accept can't fire on the same cycle because
            // s_axis_tready=0 when have_subblock=1.)
            if (consume) begin
                have_subblock <= 1'b0;
                phase         <= 3'd0;
            end

            // Latch the incoming beat into its slot.
            if (beat_accept) begin
                case (phase)
                    3'd0: weight_bits_r       <= s_axis_tdata[31:0];
                    3'd1: acts_r[63:0]        <= s_axis_tdata;
                    3'd2: acts_r[127:64]      <= s_axis_tdata;
                    3'd3: acts_r[191:128]     <= s_axis_tdata;
                    3'd4: acts_r[255:192]     <= s_axis_tdata;
                    3'd5: begin
                        weight_scale_r <= s_axis_tdata[15:0];
                        act_scale_r    <= s_axis_tdata[31:16];
                    end
                    default: ;
                endcase

                if (phase == 3'd5)
                    have_subblock <= 1'b1;
                else
                    phase <= phase + 3'd1;
            end
        end
    end
endmodule
