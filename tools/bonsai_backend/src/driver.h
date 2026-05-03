#pragma once

#include <cstdint>
#include <memory>

// Layer 3: Bonsai command drivers.
// Drivers know the external tile protocol only. They do not know ggml tensors;
// they just execute CLEAR/LDW/LDA/SEED/START/RDP against DriverKind.

namespace bonsai {

struct Tile {
    static constexpr int rows = 2;
    static constexpr int cols = 2;
    static constexpr int act_bits = 8;
    static constexpr int psum_bits = 16;
};

enum Status : uint8_t {
    status_busy        = 1u << 0,
    status_done        = 1u << 1,
    status_weight_done = 1u << 2,
    status_all_valid   = 1u << 3,
    status_start_ready = 1u << 4,
    status_idle_stable = 1u << 5,
    status_error       = 1u << 6,
};

enum class DriverKind {
    cpu,
    verilator,
    asic,
};

class BonsaiDriver {
public:
    virtual ~BonsaiDriver() = default;

    virtual const char * name() const = 0;

    virtual void clear() = 0;
    virtual void ldw(int row, uint8_t packed_weights) = 0;
    virtual void lda(int col, int8_t act) = 0;
    virtual void seed(int row, int16_t psum) = 0;
    virtual void start() = 0;
    virtual uint8_t status() = 0;
    virtual int16_t rdp(int row) = 0;

    // Run one K-tile (clear → ldw → lda × Tile::cols → seed → start → wait
    // → rdp) and return the row-0 psum. `acts` points to Tile::cols values.
    // Default impl walks the per-command interface above; the asic driver
    // overrides to pack the whole sequence into one USB X-frame, removing
    // ~10 round-trips per tile. Errors are signaled through driver state
    // (status_error bit) rather than the return value, so callers should
    // do one status() check at the end of a matmul rather than per-tile.
    virtual int16_t run_tile(uint8_t packed_weights,
                             const int8_t * acts,
                             int16_t seed_value);

    // Run `n` K-tiles. Inputs are flat arrays:
    //   packed_weights[i] for tile i
    //   acts[i * Tile::cols + lane] for tile i, lane
    //   seeds[i] for tile i (may be nullptr → all zeros)
    //   psums_out[i] receives the row-0 psum for tile i
    // Default impl loops `run_tile` serially. The asic driver overrides to
    // pipeline N USB transfers concurrently, eliminating the per-tile
    // round-trip stall (one URB outstanding caps at ~250 µs/tile on USB FS;
    // pipelining gets us close to the wire-rate ~50 µs/tile ceiling).
    virtual void run_tile_batch(const uint8_t * packed_weights,
                                const int8_t  * acts,
                                const int16_t * seeds,
                                int16_t       * psums_out,
                                int n);
};

const char * driver_kind_name(DriverKind kind);
DriverKind driver_kind_from_env();

std::unique_ptr<BonsaiDriver> create_bonsai_driver(DriverKind kind);
std::unique_ptr<BonsaiDriver> create_cpu_bonsai_driver();
std::unique_ptr<BonsaiDriver> create_verilator_bonsai_driver();
std::unique_ptr<BonsaiDriver> create_asic_bonsai_driver();

} // namespace bonsai
