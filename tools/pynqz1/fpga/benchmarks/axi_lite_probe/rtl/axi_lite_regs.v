// axi_lite_regs - hand-written AXI4-Lite slave with a small register file.
//
// Used as the proof-of-plumbing for the W1A8 control plane: if PYNQ can read
// the magic constants, round-trip the scratch reg, and watch the counter
// advance at the expected rate, then the same path will carry start/done/
// address registers for a real compute kernel.
//
// Register map (byte offsets, 32-bit aligned):
//   0x00  ID       RO  0xCAFE_0001   magic constant
//   0x04  VERSION  RO  0x0000_0001   bumped if this map changes
//   0x08  SCRATCH  RW                any value the host writes
//   0x0C  CTRL     RW                bit[0]=run; bit[1]=reset-counter strobe
//   0x10  COUNTER  RO                free-running counter while CTRL[0]=1
//
// Writes to RO regs are silently dropped (response OKAY, no state change),
// matching standard AXI-Lite practice.
//
// Wstrb is honored byte-by-byte. Reads always return all 32 bits.

`default_nettype none

module axi_lite_regs #(
    parameter integer ADDR_WIDTH = 8,
    parameter integer DATA_WIDTH = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input  wire                       s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                       s_axi_aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input  wire [ADDR_WIDTH-1:0]      s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0]                 s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                       s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire                       s_axi_awready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [DATA_WIDTH-1:0]      s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [(DATA_WIDTH/8)-1:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                       s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire                       s_axi_wready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]                 s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire                       s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                       s_axi_bready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [ADDR_WIDTH-1:0]      s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire [2:0]                 s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire                       s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire                       s_axi_arready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [DATA_WIDTH-1:0]      s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]                 s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire                       s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire                       s_axi_rready
);

    // -- Register map (decoded address bits) -----------------------------
    // Only bits [4:2] are decoded — the lower 2 are byte-within-word and
    // the upper bits are interconnect-decoded, so 5 regs need 3 decode bits.
    localparam [4:0] ADDR_ID      = 5'h00;
    localparam [4:0] ADDR_VERSION = 5'h04;
    localparam [4:0] ADDR_SCRATCH = 5'h08;
    localparam [4:0] ADDR_CTRL    = 5'h0C;
    localparam [4:0] ADDR_COUNTER = 5'h10;

    localparam [31:0] ID_VALUE      = 32'hCAFE_0001;
    localparam [31:0] VERSION_VALUE = 32'h0000_0001;

    // -- Storage ---------------------------------------------------------
    reg [31:0] scratch_q;
    reg [31:0] ctrl_q;
    reg [31:0] counter_q;

    // -- Write channel ---------------------------------------------------
    // Textbook AXI-Lite slave: hold *ready low until both AW and W arrive,
    // then accept both on the same cycle and raise BVALID next cycle.
    reg awready_q, wready_q, bvalid_q;
    reg [ADDR_WIDTH-1:0] awaddr_q;

    wire write_accept =
        !awready_q && !wready_q && s_axi_awvalid && s_axi_wvalid && !bvalid_q;
    wire write_commit = awready_q && wready_q;  // pulses for one cycle

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            awready_q <= 1'b0;
            wready_q  <= 1'b0;
            bvalid_q  <= 1'b0;
            awaddr_q  <= {ADDR_WIDTH{1'b0}};
        end else begin
            // *ready pulse: one cycle when both channels valid.
            awready_q <= write_accept;
            wready_q  <= write_accept;
            if (write_accept) begin
                awaddr_q <= s_axi_awaddr;
            end

            // BVALID rises after the accept and falls when host acks.
            if (write_commit) begin
                bvalid_q <= 1'b1;
            end else if (bvalid_q && s_axi_bready) begin
                bvalid_q <= 1'b0;
            end
        end
    end

    // Byte-strobe-aware write merge.
    function [31:0] apply_wstrb;
        input [31:0] current;
        input [31:0] wdata;
        input [3:0]  wstrb;
        begin
            apply_wstrb = {
                wstrb[3] ? wdata[31:24] : current[31:24],
                wstrb[2] ? wdata[23:16] : current[23:16],
                wstrb[1] ? wdata[15:8]  : current[15:8],
                wstrb[0] ? wdata[7:0]   : current[7:0]
            };
        end
    endfunction

    // -- Register writes -------------------------------------------------
    wire ctrl_write = write_commit && (awaddr_q[4:0] == ADDR_CTRL);
    wire counter_clear_strobe = ctrl_write && s_axi_wdata[1] && s_axi_wstrb[0];

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            scratch_q <= 32'h0;
            ctrl_q    <= 32'h0;
        end else if (write_commit) begin
            case (awaddr_q[4:0])
                ADDR_SCRATCH: scratch_q <= apply_wstrb(scratch_q, s_axi_wdata, s_axi_wstrb);
                ADDR_CTRL:    ctrl_q    <= apply_wstrb(ctrl_q,    s_axi_wdata, s_axi_wstrb);
                default: /* RO or unmapped: drop silently */ ;
            endcase
        end
    end

    // -- Free-running counter -------------------------------------------
    // Reset strobe (CTRL.bit[1]) takes precedence over the increment so a
    // single write of 0x03 simultaneously clears and starts. The strobe is
    // *not* stored in ctrl_q (only bit[0] is meaningful between writes).
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            counter_q <= 32'h0;
        end else if (counter_clear_strobe) begin
            counter_q <= 32'h0;
        end else if (ctrl_q[0]) begin
            counter_q <= counter_q + 32'h1;
        end
    end

    // -- Read channel ---------------------------------------------------
    reg arready_q, rvalid_q;
    reg [31:0] rdata_q;

    wire read_accept = !arready_q && s_axi_arvalid && !rvalid_q;

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            arready_q <= 1'b0;
            rvalid_q  <= 1'b0;
            rdata_q   <= 32'h0;
        end else begin
            arready_q <= read_accept;

            if (read_accept) begin
                rvalid_q <= 1'b1;
                case (s_axi_araddr[4:0])
                    ADDR_ID:      rdata_q <= ID_VALUE;
                    ADDR_VERSION: rdata_q <= VERSION_VALUE;
                    ADDR_SCRATCH: rdata_q <= scratch_q;
                    ADDR_CTRL:    rdata_q <= ctrl_q;
                    ADDR_COUNTER: rdata_q <= counter_q;
                    default:      rdata_q <= 32'h0;
                endcase
            end else if (rvalid_q && s_axi_rready) begin
                rvalid_q <= 1'b0;
            end
        end
    end

    // -- Outputs --------------------------------------------------------
    assign s_axi_awready = awready_q;
    assign s_axi_wready  = wready_q;
    assign s_axi_bvalid  = bvalid_q;
    assign s_axi_bresp   = 2'b00;       // OKAY
    assign s_axi_arready = arready_q;
    assign s_axi_rvalid  = rvalid_q;
    assign s_axi_rdata   = rdata_q;
    assign s_axi_rresp   = 2'b00;       // OKAY

    // Silence lint on unused AXI signals we don't model (PROT bits).
    wire _unused = &{s_axi_awprot, s_axi_arprot, 1'b0};

endmodule
