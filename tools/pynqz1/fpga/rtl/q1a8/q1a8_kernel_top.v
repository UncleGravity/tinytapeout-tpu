// q1a8_kernel_top - synthesizable Vivado top for the W1A8 matmul kernel.
//
// Wraps q1a8_kernel with a small AXI4-Lite slave register file and exposes:
//   - AXI4-Lite control plane (host reads/writes registers via the PS GP0)
//   - AXI4-Stream data input (one Q8 sub-block per 6 beats from an AXI DMA)
//
// Register map (32-bit aligned, byte offsets):
//   0x00  ID             RO  0xB05A_1000
//   0x04  VERSION        RO  0x0000_0001
//   0x08  CTRL           RW  bit[0] = start-kernel strobe (writes only)
//   0x0C  STATUS         RO  bit[0] = busy, bit[1] = done_latched
//   0x10  NUM_SUBBLOCKS  RW  how many Q8 sub-blocks for the next kernel
//   0x14  RESULT         RO  fp32 accumulator from the last kernel
//   0x18  CYCLES         RO  cycle count of the last kernel (perf counter)
//
// `done_latched` captures the 1-cycle `kernel_done` pulse so the host can
// poll it without missing the event. It clears on the next start strobe.
//
// AXI-Lite slave is hand-rolled here (same handshake pattern as the
// axi_lite_probe). When a third bitstream needs an AXI-Lite slave it'll be
// worth extracting a generic skeleton to rtl/common/.

`default_nettype none

module q1a8_kernel_top (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input  wire         s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire         s_axi_aresetn,

    // -- AXI4-Lite slave (control) -----------------------------------------
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR"  *) input  wire [7:0]   s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT"  *) input  wire [2:0]   s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *) input  wire         s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *) output wire         s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA"   *) input  wire [31:0]  s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB"   *) input  wire [3:0]   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID"  *) input  wire         s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY"  *) output wire         s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP"   *) output wire [1:0]   s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID"  *) output wire         s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY"  *) input  wire         s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR"  *) input  wire [7:0]   s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT"  *) input  wire [2:0]   s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *) input  wire         s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *) output wire         s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA"   *) output wire [31:0]  s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP"   *) output wire [1:0]   s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID"  *) output wire         s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY"  *) input  wire         s_axi_rready,

    // -- AXI4-Stream slave (data) ------------------------------------------
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA"  *) input  wire [63:0]  s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input  wire         s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire         s_axis_tready
);
    wire clk   = s_axi_aclk;
    wire rst_n = s_axi_aresetn;

    // -- Register map constants -----------------------------------------
    localparam [4:0] ADDR_ID            = 5'h00;
    localparam [4:0] ADDR_VERSION       = 5'h04;
    localparam [4:0] ADDR_CTRL          = 5'h08;
    localparam [4:0] ADDR_STATUS        = 5'h0C;
    localparam [4:0] ADDR_NUM_SUBBLOCKS = 5'h10;
    localparam [4:0] ADDR_RESULT        = 5'h14;
    localparam [4:0] ADDR_CYCLES        = 5'h18;

    localparam [31:0] ID_VALUE      = 32'hB05A_1000;
    localparam [31:0] VERSION_VALUE = 32'h0000_0001;

    // -- Control storage ------------------------------------------------
    reg [15:0] num_subblocks_q;
    reg        start_strobe;     // 1-cycle pulse; not stored persistently
    reg        done_latched;
    reg [31:0] cycle_count_q;

    // -- Kernel instance -------------------------------------------------
    wire        kernel_busy;
    wire        kernel_done;
    wire [31:0] kernel_result;

    q1a8_kernel u_kernel (
        .clk(clk), .rst_n(rst_n),
        .start_kernel(start_strobe),
        .num_subblocks(num_subblocks_q),
        .kernel_done(kernel_done),
        .busy(kernel_busy),
        .result(kernel_result),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready)
    );

    // -- done_latched: capture the 1-cycle pulse; clear on next start ---
    always @(posedge clk) begin
        if (!rst_n)            done_latched <= 1'b0;
        else if (start_strobe) done_latched <= 1'b0;
        else if (kernel_done)  done_latched <= 1'b1;
    end

    // -- Cycle counter (free profiling) --------------------------------
    always @(posedge clk) begin
        if (!rst_n)            cycle_count_q <= 32'd0;
        else if (start_strobe) cycle_count_q <= 32'd0;
        else if (kernel_busy)  cycle_count_q <= cycle_count_q + 32'd1;
    end

    // -- AXI4-Lite handshake (write path) -------------------------------
    reg awready_q, wready_q, bvalid_q;
    reg [7:0] awaddr_q;

    wire write_accept =
        !awready_q && !wready_q && s_axi_awvalid && s_axi_wvalid && !bvalid_q;
    wire write_commit = awready_q && wready_q;

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

    always @(posedge clk) begin
        if (!rst_n) begin
            awready_q       <= 1'b0;
            wready_q        <= 1'b0;
            bvalid_q        <= 1'b0;
            awaddr_q        <= 8'd0;
            num_subblocks_q <= 16'd0;
            start_strobe    <= 1'b0;
        end else begin
            // start_strobe is always a single-cycle pulse; default low.
            start_strobe <= 1'b0;

            awready_q <= write_accept;
            wready_q  <= write_accept;
            if (write_accept) awaddr_q <= s_axi_awaddr;

            if (write_commit)                       bvalid_q <= 1'b1;
            else if (bvalid_q && s_axi_bready)      bvalid_q <= 1'b0;

            // Register writes (RO regs silently drop).
            if (write_commit) begin
                case (awaddr_q[4:0])
                    ADDR_CTRL: begin
                        // bit[0] is a write-only strobe (does not latch into a reg).
                        if (s_axi_wstrb[0] && s_axi_wdata[0])
                            start_strobe <= 1'b1;
                    end
                    ADDR_NUM_SUBBLOCKS: begin
                        if (s_axi_wstrb[0]) num_subblocks_q[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) num_subblocks_q[15:8] <= s_axi_wdata[15:8];
                    end
                    default: /* RO or unmapped: drop silently */ ;
                endcase
            end
        end
    end

    // -- AXI4-Lite handshake (read path) --------------------------------
    reg        arready_q, rvalid_q;
    reg [31:0] rdata_q;

    wire read_accept = !arready_q && s_axi_arvalid && !rvalid_q;

    always @(posedge clk) begin
        if (!rst_n) begin
            arready_q <= 1'b0;
            rvalid_q  <= 1'b0;
            rdata_q   <= 32'd0;
        end else begin
            arready_q <= read_accept;
            if (read_accept) begin
                rvalid_q <= 1'b1;
                case (s_axi_araddr[4:0])
                    ADDR_ID:            rdata_q <= ID_VALUE;
                    ADDR_VERSION:       rdata_q <= VERSION_VALUE;
                    ADDR_CTRL:          rdata_q <= 32'd0;             // strobe reads as 0
                    ADDR_STATUS:        rdata_q <= {30'd0, done_latched, kernel_busy};
                    ADDR_NUM_SUBBLOCKS: rdata_q <= {16'd0, num_subblocks_q};
                    ADDR_RESULT:        rdata_q <= kernel_result;
                    ADDR_CYCLES:        rdata_q <= cycle_count_q;
                    default:            rdata_q <= 32'd0;
                endcase
            end else if (rvalid_q && s_axi_rready) begin
                rvalid_q <= 1'b0;
            end
        end
    end

    // -- Outputs -------------------------------------------------------
    assign s_axi_awready = awready_q;
    assign s_axi_wready  = wready_q;
    assign s_axi_bvalid  = bvalid_q;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = arready_q;
    assign s_axi_rvalid  = rvalid_q;
    assign s_axi_rdata   = rdata_q;
    assign s_axi_rresp   = 2'b00;

    wire _unused = &{s_axi_awprot, s_axi_arprot, 1'b0};
endmodule
