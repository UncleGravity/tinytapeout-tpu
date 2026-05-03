#include "driver.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

int main() {
    const bonsai::DriverKind kind = bonsai::driver_kind_from_env();
    std::unique_ptr<bonsai::BonsaiDriver> driver = bonsai::create_bonsai_driver(kind);
    if (driver == nullptr) {
        std::fprintf(stderr, "driver-bench: requested driver is not available\n");
        return 1;
    }

    constexpr int n_iters = 1'000'000;

    // Mix of the protocol calls a real matmul issues per K-tile so the rate
    // reflects what you'll actually see in inference, not just status polls.
    // Per loop body: clear (2 cyc) + ldw + lda + lda + seed (2 cyc) + start
    // (2 cyc) + status + rdp (2 cyc) = 12 simulated clock cycles.
    constexpr int cycles_per_iter = 12;

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    int16_t sink = 0;
    for (int i = 0; i < n_iters; ++i) {
        driver->clear();
        driver->ldw(0, (uint8_t) (i & 0xff));
        driver->lda(0, (int8_t) (i & 0x7f));
        driver->lda(1, (int8_t) ((i >> 1) & 0x7f));
        driver->seed(0, 0);
        driver->start();
        (void) driver->status();
        sink ^= driver->rdp(0);
    }

    const auto t1 = clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    const double iters_per_sec = (double) n_iters / secs;
    const double cycles_per_sec = iters_per_sec * cycles_per_iter;

    std::printf(
        "driver-bench: driver=%s iters=%d  %.2f s  %.3f Miters/s  ~%.2f MHz simulated  (sink=%d)\n",
        driver->name(),
        n_iters,
        secs,
        iters_per_sec / 1e6,
        cycles_per_sec / 1e6,
        (int) sink);
    return 0;
}
