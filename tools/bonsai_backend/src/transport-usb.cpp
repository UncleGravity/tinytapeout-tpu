#include "transport.h"
#include "protocol.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <libusb.h>

// Talks to the RP2350 firmware over a USB vendor class with two raw bulk endpoints.
// The firmware exposes an 'X' (transact) mode that takes 2*N bytes [ui_in, uio_in]
// in and replies with N bytes (uo_out sampled after each clock edge).
//
// One chip per process, but llama.cpp creates several backends concurrently
// (e.g. an async-upload backend during model load alongside the compute
// backend). All UsbTransport wrappers in a process share a single
// AsicConnection (one libusb handle, one mutex). The connection lives as
// long as any wrapper holds a shared_ptr to it. The last drop closes the
// device.

namespace bonsai {

namespace {

constexpr uint16_t default_vid = 0x2E8A;
constexpr uint16_t default_pid = 0x4002;
constexpr uint8_t  bulk_ep_out = 0x01;
constexpr uint8_t  bulk_ep_in  = 0x81;
constexpr unsigned bulk_timeout_ms = 5000;

// Honor BONSAI_ASIC_VID_PID="2e8a:4002" if set; otherwise use the firmware's
// default IDs.
struct VidPid { uint16_t vid; uint16_t pid; };
VidPid resolve_vid_pid() {
    if (const char * env = std::getenv("BONSAI_ASIC_VID_PID");
            env != nullptr && env[0] != '\0') {
        unsigned int vid = 0, pid = 0;
        if (std::sscanf(env, "%x:%x", &vid, &pid) == 2 && vid <= 0xffff && pid <= 0xffff) {
            return { (uint16_t) vid, (uint16_t) pid };
        }
        std::fprintf(stderr,
            "usb transport: BONSAI_ASIC_VID_PID=%s is not 'VID:PID' hex; "
            "using default %04x:%04x\n", env, default_vid, default_pid);
    }
    return { default_vid, default_pid };
}

// Owns the libusb_context + handle and serializes all bulk traffic. Created
// lazily by create_usb_transport() and shared via weak_ptr so concurrent
// backends in the same process see one device.
struct AsicConnection {
    libusb_context *       ctx    = nullptr;
    libusb_device_handle * handle = nullptr;
    int                    iface  = 0;
    VidPid                 ids{};
    std::mutex             io_mutex;
    bool                   reset_done = false;
    bool                   reset_ok   = false;

    ~AsicConnection() {
        if (handle != nullptr) {
            libusb_release_interface(handle, iface);
            libusb_close(handle);
        }
        if (ctx != nullptr) {
            libusb_exit(ctx);
        }
    }
};

std::mutex g_conn_mutex;
std::weak_ptr<AsicConnection> g_conn_weak;

bool bulk_write(AsicConnection & conn, const void * buf, size_t n) {
    uint8_t * p = const_cast<uint8_t *>((const uint8_t *) buf);
    size_t   off = 0;
    while (off < n) {
        int transferred = 0;
        const int rc = libusb_bulk_transfer(
            conn.handle, bulk_ep_out, p + off,
            (int) (n - off), &transferred, bulk_timeout_ms);
        if (rc != LIBUSB_SUCCESS) {
            std::fprintf(stderr, "usb transport: bulk write failed: %s\n",
                libusb_error_name(rc));
            return false;
        }
        off += (size_t) transferred;
    }
    return true;
}

bool bulk_read(AsicConnection & conn, void * buf, size_t n) {
    uint8_t * p = (uint8_t *) buf;
    size_t   off = 0;
    while (off < n) {
        int transferred = 0;
        const int rc = libusb_bulk_transfer(
            conn.handle, bulk_ep_in, p + off,
            (int) (n - off), &transferred, bulk_timeout_ms);
        if (rc != LIBUSB_SUCCESS) {
            std::fprintf(stderr, "usb transport: bulk read failed: %s\n",
                libusb_error_name(rc));
            return false;
        }
        off += (size_t) transferred;
    }
    return true;
}

// Wrap N (ui, uio) pairs in a 5-byte X-frame header; send + receive in one
// libusb round-trip. Caller holds conn.io_mutex.
std::vector<uint8_t> x_frame_locked(
        AsicConnection & conn,
        const std::vector<std::pair<uint8_t, uint8_t>> & pairs) {
    const uint32_t n = (uint32_t) pairs.size();
    std::vector<uint8_t> tx(5 + 2 * (size_t) n);
    tx[0] = 'X';
    tx[1] = (uint8_t) (n & 0xff);
    tx[2] = (uint8_t) ((n >> 8)  & 0xff);
    tx[3] = (uint8_t) ((n >> 16) & 0xff);
    tx[4] = (uint8_t) ((n >> 24) & 0xff);
    for (uint32_t i = 0; i < n; ++i) {
        tx[5 + 2 * i]     = pairs[i].first;
        tx[5 + 2 * i + 1] = pairs[i].second;
    }
    if (!bulk_write(conn, tx.data(), tx.size())) return {};
    std::vector<uint8_t> resp(n);
    if (!bulk_read(conn, resp.data(), resp.size())) resp.clear();
    return resp;
}

class UsbTransport final : public Transport {
public:
    UsbTransport(std::shared_ptr<AsicConnection> conn_, bool dead_) :
        conn(std::move(conn_)),
        dead(dead_) {}

    UsbTransport(const UsbTransport &) = delete;
    UsbTransport & operator=(const UsbTransport &) = delete;

    const char * name() const override { return "usb"; }

    // Pack the entire Plan into ONE X-frame: 5-byte header + N * tile_cycles
    // (ui, uio) pairs. The chip's per-cycle commands handle clear/start/rdp
    // for each tile within the batch — a single USB round-trip drives N
    // chip tiles. Two reasons we don't use libusb_submit_transfer pipelining
    // instead:
    //   1. The firmware's tud_vendor_write_flush is "soft" — it marks the
    //      packet boundary but does not block. With back-to-back X-frames,
    //      the firmware concatenates two flushes' worth of bytes into one
    //      bulk packet, which overflows the host's per-X-frame IN URB.
    //      Putting everything in one X-frame sidesteps the buffering issue.
    //   2. Empirically, the wire spends most of its time on per-frame
    //      kernel scheduling rather than bytes-on-the-wire. Concatenating
    //      bytes into one frame saves N - 1 round-trip stalls.
    bool execute(const Plan & plan, int16_t * outputs) override {
        const size_t n = plan.ops.size();
        const size_t out_count = n * (size_t) Tile::rows;
        if (n == 0) return true;
        if (dead) {
            std::memset(outputs, 0, out_count * sizeof(int16_t));
            return false;
        }

        // Pass 1: compute total wire cycles. Tile size is variable now —
        // run-internal tiles drop CLEAR/SEED/RDP, so each op contributes a
        // different amount.
        uint32_t total_cycles = 0;
        for (size_t i = 0; i < n; ++i) {
            total_cycles += (uint32_t) tile_cycles_for(plan.ops[i].starts_run,
                                                       plan.ops[i].ends_run);
        }

        std::vector<uint8_t> tx(5 + 2 * (size_t) total_cycles);
        std::vector<uint8_t> rx((size_t) total_cycles);
        tx[0] = 'X';
        tx[1] = (uint8_t) ( total_cycles        & 0xff);
        tx[2] = (uint8_t) ((total_cycles >>  8) & 0xff);
        tx[3] = (uint8_t) ((total_cycles >> 16) & 0xff);
        tx[4] = (uint8_t) ((total_cycles >> 24) & 0xff);

        // Pass 2: pack tiles, remembering per-op rx offsets for RDP results.
        // Ops that don't end a run leave their rdp_offset == SIZE_MAX, and
        // their psums are filled in from the run-tail tile that follows.
        std::vector<size_t> rdp_offset(n, (size_t) -1);
        size_t cycle = 0;
        size_t pending_run_start = (size_t) -1;
        for (size_t i = 0; i < n; ++i) {
            const PlanOp & op = plan.ops[i];
            const MatmulTileAttrs & m = op.attrs.matmul_tile;
            int rdp_within = -1;
            const int tile_n = build_tile_pairs(
                &tx[5 + 2 * cycle],
                m.packed_weights, m.acts, m.seeds,
                op.starts_run, op.ends_run,
                &rdp_within);
            if (op.starts_run) {
                pending_run_start = i;
            }
            if (op.ends_run && rdp_within >= 0) {
                // Every op since pending_run_start writes its psums from
                // this tail tile's RDP. The chip's acc_q only holds the
                // FINAL accumulated psum at the run tail, so non-tail ops
                // in a multi-tile run get the same psum value (the run's
                // total). matmul.cpp accumulates pre-run, so it only
                // reads outputs[run_tail * rows + r] in practice.
                rdp_offset[i] = cycle + (size_t) rdp_within;
            }
            cycle += (size_t) tile_n;
            (void) pending_run_start;
        }

        std::lock_guard<std::mutex> guard(conn->io_mutex);
        if (!bulk_write(*conn, tx.data(), tx.size()) ||
            !bulk_read (*conn, rx.data(), rx.size())) {
            dead = true;
            std::memset(outputs, 0, out_count * sizeof(int16_t));
            return false;
        }

        for (size_t i = 0; i < n; ++i) {
            if (rdp_offset[i] == (size_t) -1) {
                // Tile didn't end a run, no RDP results for it. Zero the
                // outputs so callers reading these slots don't see junk.
                for (int row = 0; row < Tile::rows; ++row) {
                    outputs[i * (size_t) Tile::rows + row] = 0;
                }
                continue;
            }
            for (int row = 0; row < Tile::rows; ++row) {
                outputs[i * (size_t) Tile::rows + row] =
                    parse_psum_at(rx.data(), rdp_offset[i], row);
            }
        }
        return true;
    }

    uint8_t status() override {
        if (dead) return status_error;
        std::lock_guard<std::mutex> guard(conn->io_mutex);
        const auto resp = x_frame_locked(*conn, {{pack_ui(cmd_status), 0}});
        if (resp.empty()) {
            dead = true;
            return status_error;
        }
        return resp[0];
    }

    static bool cycle_reset_and_check(AsicConnection & conn) {
        std::lock_guard<std::mutex> guard(conn.io_mutex);
        const uint8_t assert_hdr[5]  = { 'a', 0, 0, 0, 0 };
        const uint8_t release_hdr[5] = { 'd', 0, 0, 0, 0 };
        if (!bulk_write(conn, assert_hdr, sizeof(assert_hdr))) return false;
        x_frame_locked(conn, std::vector<std::pair<uint8_t, uint8_t>>(8, {pack_ui(cmd_nop), 0}));
        if (!bulk_write(conn, release_hdr, sizeof(release_hdr))) return false;
        x_frame_locked(conn, std::vector<std::pair<uint8_t, uint8_t>>(4, {pack_ui(cmd_nop), 0}));
        const auto resp = x_frame_locked(conn, {{pack_ui(cmd_status), 0}});
        if (resp.empty()) return false;
        const uint8_t st = resp[0];
        // 0xff means uo_out pads pulled high but undriven — bitstream not
        // loaded. After reset we want IDLE/START_READY high and ERROR low.
        if (st == 0xff) return false;
        if (st & status_error) return false;
        if (!(st & status_start_ready)) return false;
        return true;
    }

private:
    std::shared_ptr<AsicConnection> conn;
    bool dead;
};

bool open_device(AsicConnection & conn) {
    int rc = libusb_init(&conn.ctx);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "usb transport: libusb_init failed: %s\n",
            libusb_error_name(rc));
        return false;
    }

    conn.handle = libusb_open_device_with_vid_pid(conn.ctx, conn.ids.vid, conn.ids.pid);
    if (conn.handle == nullptr) {
        std::fprintf(stderr,
            "usb transport: no USB device with %04x:%04x found "
            "(set BONSAI_ASIC_VID_PID to override; flash firmware/build/rp_echo.uf2)\n",
            conn.ids.vid, conn.ids.pid);
        return false;
    }

    rc = libusb_claim_interface(conn.handle, conn.iface);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "usb transport: claim_interface(%d) failed: %s\n",
            conn.iface, libusb_error_name(rc));
        return false;
    }

    // The firmware's protocol loop has no resync sentinel: if a previous
    // host disconnected after sending an X-frame header but before all body
    // bytes, the firmware is still mid-loop reading body. The next host's
    // 'a'/'d'/'X' headers would land in that stale body slot, the device
    // sends way more response bytes than we ask for, and bulk_read returns
    // LIBUSB_ERROR_OVERFLOW. libusb_reset_device forces a USB port reset
    // which restarts the firmware's main loop on a clean state machine.
    // Handle becomes invalid after the reset on some platforms, so re-open.
    libusb_reset_device(conn.handle);
    libusb_release_interface(conn.handle, conn.iface);
    libusb_close(conn.handle);

    conn.handle = libusb_open_device_with_vid_pid(conn.ctx, conn.ids.vid, conn.ids.pid);
    if (conn.handle == nullptr) {
        std::fprintf(stderr, "usb transport: device disappeared after reset\n");
        return false;
    }
    rc = libusb_claim_interface(conn.handle, conn.iface);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "usb transport: re-claim after reset failed: %s\n",
            libusb_error_name(rc));
        return false;
    }
    return true;
}

} // namespace

std::unique_ptr<Transport> create_usb_transport() {
    std::lock_guard<std::mutex> guard(g_conn_mutex);

    std::shared_ptr<AsicConnection> conn = g_conn_weak.lock();
    if (conn == nullptr) {
        conn = std::make_shared<AsicConnection>();
        conn->ids = resolve_vid_pid();
        if (!open_device(*conn)) {
            return nullptr;
        }
        g_conn_weak = conn;
    }

    if (!conn->reset_done) {
        conn->reset_ok = UsbTransport::cycle_reset_and_check(*conn);
        conn->reset_done = true;
        if (!conn->reset_ok) {
            std::fprintf(stderr,
                "usb transport: chip did not report START_READY after reset on %04x:%04x; "
                "is the bitstream loaded and the firmware in 'X' mode?\n",
                conn->ids.vid, conn->ids.pid);
        }
    }

    return std::unique_ptr<Transport>(new UsbTransport(conn, !conn->reset_ok));
}

} // namespace bonsai
