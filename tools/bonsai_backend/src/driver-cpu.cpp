#include "driver.h"

#include <algorithm>
#include <array>

namespace bonsai {

// CPU model of the observable RTL protocol. This is the software reference for
// the command stream that Verilator and ASIC drivers will execute externally.
class CpuBonsaiDriver final : public BonsaiDriver {
public:
    const char * name() const override {
        return "cpu";
    }

    void clear() override {
        act_mem.fill(0);
        acc.fill(0);
        acc_done.fill(false);
        busy_latched = false;
        done_latched = false;
        weight_done_latched = false;
        error_latched = false;
    }

    void ldw(int row, uint8_t packed_weights) override {
        if (!require_idle() || row < 0 || row >= Tile::rows) {
            error_latched = true;
            return;
        }

        for (int col = 0; col < Tile::cols; ++col) {
            weight_mem[row][col] = ((packed_weights >> col) & 1u) != 0;
        }
        weight_done_latched = false;
    }

    void lda(int col, int8_t act) override {
        if (!require_idle() || col < 0 || col >= Tile::cols) {
            error_latched = true;
            return;
        }

        act_mem[col] = act;
    }

    void seed(int row, int16_t psum) override {
        if (!require_idle() || row < 0 || row >= Tile::rows) {
            error_latched = true;
            return;
        }

        acc[row] = psum;
    }

    void start() override {
        if (!require_idle()) {
            error_latched = true;
            return;
        }

        busy_latched = true;
        done_latched = false;
        acc_done.fill(false);

        for (int row = 0; row < Tile::rows; ++row) {
            int32_t psum = acc[row];
            for (int col = 0; col < Tile::cols; ++col) {
                const int32_t act = act_mem[col];
                psum += weight_mem[row][col] ? act : -act;
            }
            acc[row] = wrap_i16(psum);
            acc_done[row] = true;
        }

        busy_latched = false;
        done_latched = true;
        weight_done_latched = true;
    }

    uint8_t status() override {
        uint8_t value = 0;
        if (busy_latched) {
            value |= status_busy;
        }
        if (done_latched) {
            value |= status_done;
        }
        if (weight_done_latched) {
            value |= status_weight_done;
        }
        if (std::all_of(acc_done.begin(), acc_done.end(), [](bool done) { return done; })) {
            value |= status_all_valid;
        }
        if (!busy_latched) {
            value |= status_start_ready | status_idle_stable;
        }
        if (error_latched) {
            value |= status_error;
        }
        return value;
    }

    int16_t rdp(int row) override {
        if (row < 0 || row >= Tile::rows) {
            return 0;
        }
        return acc[row];
    }

private:
    std::array<std::array<bool, Tile::cols>, Tile::rows> weight_mem{};
    std::array<int8_t, Tile::cols> act_mem{};
    std::array<int16_t, Tile::rows> acc{};
    std::array<bool, Tile::rows> acc_done{};

    bool busy_latched = false;
    bool done_latched = false;
    bool weight_done_latched = false;
    bool error_latched = false;

    bool require_idle() const {
        return !busy_latched;
    }

    static int16_t wrap_i16(int32_t value) {
        const uint16_t raw = (uint16_t) value;
        const int32_t signed_value = (raw & 0x8000u) ? (int32_t) raw - 0x10000 : (int32_t) raw;
        return (int16_t) signed_value;
    }
};

std::unique_ptr<BonsaiDriver> create_cpu_bonsai_driver() {
    return std::unique_ptr<BonsaiDriver>(new CpuBonsaiDriver());
}

} // namespace bonsai
