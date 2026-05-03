#include "driver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <libusb.h>

// ASIC driver. Talks to the rp_streamer firmware (tools/rp_streamer/) over a
// USB vendor class with two raw bulk endpoints. The firmware exposes an 'X'
// (transact) mode that takes 2*N bytes [ui_in, uio_in] in and replies with N
// bytes (uo_out sampled after each clock edge).
//
// There is one chip per process, but llama.cpp creates several backends
// concurrently (e.g. an async-upload backend during model load alongside the
// compute backend). All AsicBonsaiDriver wrappers in a process share a single
// AsicConnection (one libusb handle, one mutex). The connection lives as long
// as any wrapper holds a shared_ptr to it; the last drop closes the device.

namespace bonsai {

namespace {

constexpr uint16_t default_vid = 0x2E8A;
constexpr uint16_t default_pid = 0x4002;
constexpr uint8_t  bulk_ep_out = 0x01;
constexpr uint8_t  bulk_ep_in  = 0x81;
constexpr unsigned bulk_timeout_ms = 5000;

enum Command : uint8_t {
    cmd_status = 0,
    cmd_clear  = 1,
    cmd_ldw    = 2,
    cmd_lda    = 3,
    cmd_seed   = 4,
    cmd_start  = 5,
    cmd_rdp    = 6,
    cmd_nop    = 7,
};

constexpr int bit_width_for_count(int count) {
    int bits = 1;
    int max_value = count - 1;
    while ((1 << bits) <= max_value) {
        ++bits;
    }
    return bits;
}

constexpr int row_bits = bit_width_for_count(Tile::rows);
constexpr int col_bits = bit_width_for_count(Tile::cols);
constexpr int psum_bytes = (Tile::psum_bits + 7) / 8;

uint8_t pack_ui(Command cmd, uint8_t arg = 0) {
    return (uint8_t) cmd | (uint8_t) ((arg & 0x1fu) << 3);
}

uint8_t encode_row_arg(int row) {
    return (uint8_t) (row & ((1 << row_bits) - 1));
}

uint8_t encode_col_arg(int col) {
    return (uint8_t) (col & ((1 << col_bits) - 1));
}

uint8_t encode_row_byte_arg(int row, int byte) {
    return (uint8_t) (encode_row_arg(row) | (byte << row_bits));
}

int16_t sign_extend_psum(uint16_t raw) {
    constexpr uint16_t sign = 1u << (Tile::psum_bits - 1);
    constexpr uint16_t mask = (Tile::psum_bits == 16) ? 0xffffu : ((1u << Tile::psum_bits) - 1u);
    raw &= mask;
    return (raw & sign) ? (int16_t) ((int32_t) raw - (int32_t) (mask + 1u)) : (int16_t) raw;
}

// Per-tile cycle layout. A "tile" is the chip-side sequence
// clear+nop / ldw / lda × cols / seed × psum_bytes / start+nop / NOP padding
// for the chip's compute latency / rdp × psum_bytes.
constexpr int tile_done_pad_cycles = 6;
constexpr int tile_cycles =
    /* clear+nop  */ 2 +
    /* ldw        */ 1 +
    /* lda × cols */ Tile::cols +
    /* seed bytes */ psum_bytes +
    /* start+nop  */ 2 +
    /* done pad   */ tile_done_pad_cycles +
    /* rdp bytes  */ psum_bytes;
constexpr int tile_rdp_offset_within_tile = tile_cycles - psum_bytes;

// Fills `dst[2*tile_cycles]` with one tile's (ui, uio) byte pairs (no
// 5-byte X-frame header — caller wraps N tiles in a single header).
inline void build_tile_pairs(uint8_t * dst,
                             uint8_t packed_weights,
                             const int8_t * acts,
                             int16_t seed_value) {
    const uint16_t seed_raw = (uint16_t) seed_value;
    int p = 0;
    auto put = [&](uint8_t a, uint8_t b) {
        dst[2 * p]     = a;
        dst[2 * p + 1] = b;
        ++p;
    };
    put(pack_ui(cmd_clear), 0);
    put(pack_ui(cmd_nop),   0);
    put(pack_ui(cmd_ldw, encode_row_arg(0)), packed_weights);
    for (int lane = 0; lane < Tile::cols; ++lane) {
        put(pack_ui(cmd_lda, encode_col_arg(lane)), (uint8_t) acts[lane]);
    }
    for (int byte = 0; byte < psum_bytes; ++byte) {
        put(pack_ui(cmd_seed, encode_row_byte_arg(0, byte)),
            (uint8_t) ((seed_raw >> (byte * 8)) & 0xffu));
    }
    put(pack_ui(cmd_start), 0);
    put(pack_ui(cmd_nop),   0);
    for (int i = 0; i < tile_done_pad_cycles; ++i) put(pack_ui(cmd_nop), 0);
    for (int byte = 0; byte < psum_bytes; ++byte) {
        put(pack_ui(cmd_rdp, encode_row_byte_arg(0, byte)), 0);
    }
}

inline int16_t parse_tile_psum_at(const uint8_t * rx, int tile_index) {
    const uint8_t * rdp = rx + (size_t) tile_index * tile_cycles + tile_rdp_offset_within_tile;
    uint16_t raw = 0;
    for (int byte = 0; byte < psum_bytes; ++byte) {
        raw |= (uint16_t) rdp[byte] << (byte * 8);
    }
    return sign_extend_psum(raw);
}

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
            "asic driver: BONSAI_ASIC_VID_PID=%s is not 'VID:PID' hex; "
            "using default %04x:%04x\n", env, default_vid, default_pid);
    }
    return { default_vid, default_pid };
}

// Owns the libusb_context + handle and serializes all bulk traffic. Created
// lazily by create_asic_bonsai_driver() and shared via weak_ptr so concurrent
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

class AsicBonsaiDriver final : public BonsaiDriver {
public:
    AsicBonsaiDriver(std::shared_ptr<AsicConnection> conn_, bool dead_) :
        conn(std::move(conn_)),
        dead(dead_) {}

    AsicBonsaiDriver(const AsicBonsaiDriver &) = delete;
    AsicBonsaiDriver & operator=(const AsicBonsaiDriver &) = delete;

    const char * name() const override { return "asic"; }

    // Once the chip has gone bad (init failed, or a transaction returned no
    // bytes), every subsequent call short-circuits. matmul.cpp's end-of-matmul
    // status() check sees status_error and aborts the matmul cleanly.
    void clear() override {
        if (dead) return;
        x_frame({{pack_ui(cmd_clear), 0}, {pack_ui(cmd_nop), 0}});
    }

    void ldw(int row, uint8_t packed_weights) override {
        if (dead) return;
        x_frame({{pack_ui(cmd_ldw, encode_row_arg(row)), packed_weights}});
    }

    void lda(int col, int8_t act) override {
        if (dead) return;
        x_frame({{pack_ui(cmd_lda, encode_col_arg(col)), (uint8_t) act}});
    }

    void seed(int row, int16_t psum) override {
        if (dead) return;
        const uint16_t raw = (uint16_t) psum;
        std::vector<std::pair<uint8_t, uint8_t>> pairs;
        pairs.reserve(psum_bytes);
        for (int byte = 0; byte < psum_bytes; ++byte) {
            pairs.push_back({
                pack_ui(cmd_seed, encode_row_byte_arg(row, byte)),
                (uint8_t) ((raw >> (byte * 8)) & 0xffu),
            });
        }
        x_frame(pairs);
    }

    void start() override {
        if (dead) return;
        x_frame({{pack_ui(cmd_start), 0}, {pack_ui(cmd_nop), 0}});
    }

    uint8_t status() override {
        if (dead) return status_error;
        const auto resp = x_frame({{pack_ui(cmd_status), 0}});
        if (resp.empty()) {
            dead = true;
            return status_error;
        }
        return resp[0];
    }

    int16_t rdp(int row) override {
        if (dead) return 0;
        std::vector<std::pair<uint8_t, uint8_t>> pairs;
        pairs.reserve(psum_bytes);
        for (int byte = 0; byte < psum_bytes; ++byte) {
            pairs.push_back({pack_ui(cmd_rdp, encode_row_byte_arg(row, byte)), 0});
        }
        const auto resp = x_frame(pairs);
        if ((int) resp.size() < psum_bytes) {
            dead = true;
            return 0;
        }
        uint16_t raw = 0;
        for (int byte = 0; byte < psum_bytes; ++byte) {
            raw |= (uint16_t) resp[byte] << (byte * 8);
        }
        return sign_extend_psum(raw);
    }

    // Run one K-tile.
    int16_t run_tile(uint8_t packed_weights,
                     const int8_t * acts,
                     int16_t seed_value) override {
        int16_t psum = 0;
        run_tile_batch(&packed_weights, acts, &seed_value, &psum, 1);
        return psum;
    }

    // Pack N K-tiles into ONE X-frame (5-byte header + N * tile_cycles
    // (ui,uio) pairs). The chip's per-cycle commands handle clear/start/rdp
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
    void run_tile_batch(const uint8_t * packed_weights,
                        const int8_t  * acts,
                        const int16_t * seeds,
                        int16_t       * psums_out,
                        int n) override {
        if (n <= 0) return;
        if (dead) {
            std::memset(psums_out, 0, (size_t) n * sizeof(int16_t));
            return;
        }

        const uint32_t total_cycles = (uint32_t) n * (uint32_t) tile_cycles;
        std::vector<uint8_t> tx(5 + 2 * (size_t) total_cycles);
        std::vector<uint8_t> rx((size_t) total_cycles);
        tx[0] = 'X';
        tx[1] = (uint8_t) ( total_cycles        & 0xff);
        tx[2] = (uint8_t) ((total_cycles >>  8) & 0xff);
        tx[3] = (uint8_t) ((total_cycles >> 16) & 0xff);
        tx[4] = (uint8_t) ((total_cycles >> 24) & 0xff);
        for (int i = 0; i < n; ++i) {
            const int16_t s = seeds ? seeds[i] : (int16_t) 0;
            build_tile_pairs(&tx[5 + (size_t) i * 2 * tile_cycles],
                             packed_weights[i],
                             acts + (size_t) i * Tile::cols, s);
        }

        std::lock_guard<std::mutex> guard(conn->io_mutex);
        if (!bulk_write(*conn, tx.data(), tx.size()) ||
            !bulk_read (*conn, rx.data(), rx.size())) {
            dead = true;
            std::memset(psums_out, 0, (size_t) n * sizeof(int16_t));
            return;
        }

        for (int i = 0; i < n; ++i) {
            psums_out[i] = parse_tile_psum_at(rx.data(), i);
        }
    }

private:
    std::shared_ptr<AsicConnection> conn;
    bool dead;

    std::vector<uint8_t> x_frame(const std::vector<std::pair<uint8_t, uint8_t>> & pairs) {
        std::lock_guard<std::mutex> guard(conn->io_mutex);
        return x_frame_locked(*conn, pairs);
    }

    static std::vector<uint8_t> x_frame_locked(
            AsicConnection & conn,
            const std::vector<std::pair<uint8_t, uint8_t>> & pairs) {
        const uint32_t n = (uint32_t) pairs.size();
        // 5-byte header followed inline by the body, so the firmware sees one
        // contiguous bulk transfer per X-frame and we don't pay a USB FS
        // frame boundary between them.
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

    static bool bulk_write(AsicConnection & conn, const void * buf, size_t n) {
        uint8_t * p = const_cast<uint8_t *>((const uint8_t *) buf);
        size_t   off = 0;
        while (off < n) {
            int transferred = 0;
            const int rc = libusb_bulk_transfer(
                conn.handle, bulk_ep_out, p + off,
                (int) (n - off), &transferred, bulk_timeout_ms);
            if (rc != LIBUSB_SUCCESS) {
                std::fprintf(stderr, "asic driver: bulk write failed: %s\n",
                    libusb_error_name(rc));
                return false;
            }
            off += (size_t) transferred;
        }
        return true;
    }

    static bool bulk_read(AsicConnection & conn, void * buf, size_t n) {
        uint8_t * p = (uint8_t *) buf;
        size_t   off = 0;
        while (off < n) {
            int transferred = 0;
            const int rc = libusb_bulk_transfer(
                conn.handle, bulk_ep_in, p + off,
                (int) (n - off), &transferred, bulk_timeout_ms);
            if (rc != LIBUSB_SUCCESS) {
                std::fprintf(stderr, "asic driver: bulk read failed: %s\n",
                    libusb_error_name(rc));
                return false;
            }
            off += (size_t) transferred;
        }
        return true;
    }

public:
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
};

bool open_device(AsicConnection & conn) {
    int rc = libusb_init(&conn.ctx);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "asic driver: libusb_init failed: %s\n",
            libusb_error_name(rc));
        return false;
    }

    conn.handle = libusb_open_device_with_vid_pid(conn.ctx, conn.ids.vid, conn.ids.pid);
    if (conn.handle == nullptr) {
        std::fprintf(stderr,
            "asic driver: no USB device with %04x:%04x found "
            "(set BONSAI_ASIC_VID_PID to override; flash firmware/build/rp_echo.uf2)\n",
            conn.ids.vid, conn.ids.pid);
        return false;
    }

    rc = libusb_claim_interface(conn.handle, conn.iface);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "asic driver: claim_interface(%d) failed: %s\n",
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
        std::fprintf(stderr, "asic driver: device disappeared after reset\n");
        return false;
    }
    rc = libusb_claim_interface(conn.handle, conn.iface);
    if (rc != LIBUSB_SUCCESS) {
        std::fprintf(stderr, "asic driver: re-claim after reset failed: %s\n",
            libusb_error_name(rc));
        return false;
    }
    return true;
}

} // namespace

std::unique_ptr<BonsaiDriver> create_asic_bonsai_driver() {
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
        conn->reset_ok = AsicBonsaiDriver::cycle_reset_and_check(*conn);
        conn->reset_done = true;
        if (!conn->reset_ok) {
            std::fprintf(stderr,
                "asic driver: chip did not report START_READY after reset on %04x:%04x; "
                "is the bitstream loaded and the firmware in 'X' mode?\n",
                conn->ids.vid, conn->ids.pid);
        }
    }

    return std::unique_ptr<BonsaiDriver>(new AsicBonsaiDriver(conn, !conn->reset_ok));
}

} // namespace bonsai
