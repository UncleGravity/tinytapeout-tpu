#include "driver.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

// ASIC driver. Talks to the rp_streamer firmware (tools/rp_streamer/) over a
// USB CDC ACM virtual serial port. The firmware exposes an 'X' (transact) mode
// that takes 2*N bytes [ui_in, uio_in] in and replies with N bytes (uo_out
// sampled after each clock edge). Each driver call lowers to one X-frame; this
// is the simplest correct mapping. Batching across calls is a follow-up.
//
// Single-instance only: there is one chip on the rp_streamer device, so a
// second create_asic_bonsai_driver() returns nullptr. matmul.cpp's worker
// thread loop already falls back to run_serial() in that case.

namespace bonsai {

namespace {

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

std::string discover_port() {
    if (const char * env = std::getenv("BONSAI_ASIC_PORT");
            env != nullptr && env[0] != '\0') {
        return env;
    }
    // macOS: /dev/tty.usbmodem*  Linux: /dev/ttyACM*
    static const char * const patterns[] = {
        "/dev/tty.usbmodem*",
        "/dev/ttyACM*",
    };
    for (const char * pattern : patterns) {
        glob_t g{};
        if (glob(pattern, 0, nullptr, &g) == 0 && g.gl_pathc > 0) {
            std::string path = g.gl_pathv[0];
            globfree(&g);
            return path;
        }
        globfree(&g);
    }
    return {};
}

std::atomic<bool> driver_open{false};

class AsicBonsaiDriver final : public BonsaiDriver {
public:
    AsicBonsaiDriver(int fd_, std::string path_) :
        fd(fd_),
        path(std::move(path_)) {
        // Pulse reset, then verify the chip lands in idle. If anything fails
        // we leave error_latched_fallback set so status() reflects it; the
        // BonsaiDriver interface is exception-free.
        if (!cycle_reset_and_check()) {
            std::fprintf(stderr,
                "asic driver: chip did not report START_READY after reset on %s; "
                "is the bitstream loaded and the firmware in 'X' mode?\n",
                path.c_str());
            dead = true;
        }
    }

    ~AsicBonsaiDriver() override {
        if (fd >= 0) {
            ::close(fd);
        }
        driver_open.store(false, std::memory_order_release);
    }

    AsicBonsaiDriver(const AsicBonsaiDriver &) = delete;
    AsicBonsaiDriver & operator=(const AsicBonsaiDriver &) = delete;

    const char * name() const override {
        return "asic";
    }

    // Once the chip has gone bad (init failed, or a transaction returned no
    // bytes), every subsequent call short-circuits to a status_error so we
    // don't keep pummeling a stuck firmware with X-frames it can't process.
    // matmul.cpp's wait_done sees status_error and aborts the matmul cleanly.
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

private:
    int fd;
    std::string path;
    bool dead = false;

    bool cycle_reset_and_check() {
        send_cmd_byte('a');  // assert reset
        x_frame(std::vector<std::pair<uint8_t, uint8_t>>(8, {pack_ui(cmd_nop), 0}));
        send_cmd_byte('d');  // release reset
        x_frame(std::vector<std::pair<uint8_t, uint8_t>>(4, {pack_ui(cmd_nop), 0}));
        const auto resp = x_frame({{pack_ui(cmd_status), 0}});
        if (resp.empty()) return false;
        const uint8_t st = resp[0];
        // 0xff means uo_out pads pulled high but undriven — bitstream not
        // loaded. After reset we want IDLE/START_READY high and ERROR low.
        if (st == 0xff) return false;
        if (st & status_error) return false;
        if (!(st & status_start_ready)) return false;
        return true;
    }

    void send_cmd_byte(char mode) {
        const uint8_t hdr[5] = { (uint8_t) mode, 0, 0, 0, 0 };
        write_all(hdr, sizeof(hdr));
    }

    std::vector<uint8_t> x_frame(const std::vector<std::pair<uint8_t, uint8_t>> & pairs) {
        const uint32_t n = (uint32_t) pairs.size();
        uint8_t hdr[5] = {
            'X',
            (uint8_t) (n & 0xff),
            (uint8_t) ((n >> 8)  & 0xff),
            (uint8_t) ((n >> 16) & 0xff),
            (uint8_t) ((n >> 24) & 0xff),
        };
        std::vector<uint8_t> body(2 * (size_t) n);
        for (uint32_t i = 0; i < n; ++i) {
            body[2 * i]     = pairs[i].first;
            body[2 * i + 1] = pairs[i].second;
        }
        if (!write_all(hdr, sizeof(hdr))) return {};
        if (!write_all(body.data(), body.size())) return {};
        std::vector<uint8_t> resp(n);
        if (!read_all(resp.data(), resp.size())) resp.clear();
        return resp;
    }

    bool write_all(const void * buf, size_t n) {
        const uint8_t * p = (const uint8_t *) buf;
        while (n > 0) {
            ssize_t w = ::write(fd, p, n);
            if (w < 0) {
                if (errno == EINTR) continue;
                std::fprintf(stderr, "asic driver: write %s failed: %s\n",
                    path.c_str(), std::strerror(errno));
                return false;
            }
            p += w;
            n -= (size_t) w;
        }
        return true;
    }

    bool read_all(void * buf, size_t n) {
        uint8_t * p = (uint8_t *) buf;
        while (n > 0) {
            ssize_t r = ::read(fd, p, n);
            if (r < 0) {
                if (errno == EINTR) continue;
                std::fprintf(stderr, "asic driver: read %s failed: %s\n",
                    path.c_str(), std::strerror(errno));
                return false;
            }
            if (r == 0) {
                std::fprintf(stderr, "asic driver: read %s EOF\n", path.c_str());
                return false;
            }
            p += r;
            n -= (size_t) r;
        }
        return true;
    }
};

int open_serial(const std::string & path) {
    int fd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        return -1;
    }
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
    }
    struct termios tio;
    if (tcgetattr(fd, &tio) == 0) {
        cfmakeraw(&tio);
        // CLOCAL: ignore modem status lines so writes don't block on DCD.
        // CREAD: enable receiver. Disable HW flow control. CDC ACM ignores
        // baud, but OS still wants something coherent set.
        tio.c_cflag |= CLOCAL | CREAD;
        tio.c_cflag &= ~CRTSCTS;
        cfsetispeed(&tio, B115200);
        cfsetospeed(&tio, B115200);
        tio.c_cc[VMIN]  = 1;
        tio.c_cc[VTIME] = 50;  // 5 s per-byte read timeout
        tcsetattr(fd, TCSANOW, &tio);
    }
    // CDC ACM on macOS expects DTR/RTS asserted before the device side will
    // see characters as flowing. pyserial does this for us; doing it manually
    // when going through raw POSIX open().
    int mbits = TIOCM_DTR | TIOCM_RTS;
    ioctl(fd, TIOCMBIS, &mbits);
    return fd;
}

} // namespace

std::unique_ptr<BonsaiDriver> create_asic_bonsai_driver() {
    bool expected = false;
    if (!driver_open.compare_exchange_strong(expected, true,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Already open. matmul.cpp will fall back to single-threaded
        // execution on the existing driver.
        return nullptr;
    }

    const std::string path = discover_port();
    if (path.empty()) {
        std::fprintf(stderr,
            "asic driver: no /dev/tty.usbmodem* / /dev/ttyACM* device found "
            "(set BONSAI_ASIC_PORT to override)\n");
        driver_open.store(false, std::memory_order_release);
        return nullptr;
    }

    const int fd = open_serial(path);
    if (fd < 0) {
        std::fprintf(stderr, "asic driver: failed to open %s: %s\n",
            path.c_str(), std::strerror(errno));
        driver_open.store(false, std::memory_order_release);
        return nullptr;
    }

    return std::unique_ptr<BonsaiDriver>(new AsicBonsaiDriver(fd, path));
}

} // namespace bonsai
