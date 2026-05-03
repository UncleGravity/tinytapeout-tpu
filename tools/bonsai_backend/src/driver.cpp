#include "driver.h"

#include <cstdlib>
#include <cstring>

namespace bonsai {

int16_t BonsaiDriver::run_tile(uint8_t packed_weights,
                               const int8_t * acts,
                               int16_t seed_value) {
    clear();
    ldw(0, packed_weights);
    for (int lane = 0; lane < Tile::cols; ++lane) {
        lda(lane, acts[lane]);
    }
    seed(0, seed_value);
    start();
    constexpr int max_polls = 128;
    for (int i = 0; i < max_polls; ++i) {
        const uint8_t st = status();
        if (st & status_error) return 0;
        if (st & status_done)  break;
    }
    return rdp(0);
}

void BonsaiDriver::run_tile_batch(const uint8_t * packed_weights,
                                  const int8_t  * acts,
                                  const int16_t * seeds,
                                  int16_t       * psums_out,
                                  int n) {
    for (int i = 0; i < n; ++i) {
        const int16_t seed_value = seeds ? seeds[i] : (int16_t) 0;
        psums_out[i] = run_tile(packed_weights[i],
                                acts + (size_t) i * Tile::cols,
                                seed_value);
    }
}

const char * driver_kind_name(DriverKind kind) {
    switch (kind) {
        case DriverKind::cpu:
            return "cpu";
        case DriverKind::verilator:
            return "verilator";
        case DriverKind::asic:
            return "asic";
    }

    return "unknown";
}

DriverKind driver_kind_from_env() {
    const char * value = std::getenv("BONSAI_DRIVER");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "cpu") == 0) {
        return DriverKind::cpu;
    }
    if (std::strcmp(value, "verilator") == 0) {
        return DriverKind::verilator;
    }
    if (std::strcmp(value, "asic") == 0) {
        return DriverKind::asic;
    }

    return DriverKind::cpu;
}

std::unique_ptr<BonsaiDriver> create_bonsai_driver(DriverKind kind) {
    switch (kind) {
        case DriverKind::cpu:
            return create_cpu_bonsai_driver();
        case DriverKind::verilator:
            return create_verilator_bonsai_driver();
        case DriverKind::asic:
            return create_asic_bonsai_driver();
    }

    return nullptr;
}

} // namespace bonsai
