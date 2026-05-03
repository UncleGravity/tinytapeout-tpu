#include "driver.h"

#include <chrono>
#include <cstdio>
#include <memory>

// Tiny diagnostic: open the ASIC driver, then time a fixed number of
// individual driver calls and report mean latency per call. Exercises the
// same code path as matmul.cpp's run_cell but with no model logic.

int main() {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("probe: creating ASIC driver...\n");
    auto driver = bonsai::create_asic_bonsai_driver();
    if (driver == nullptr) {
        std::fprintf(stderr, "probe: create_asic_bonsai_driver returned null\n");
        return 1;
    }
    std::printf("probe: driver=%s open\n", driver->name());
    std::fflush(stdout);

    // Single status read.
    auto t0 = std::chrono::steady_clock::now();
    const uint8_t st = driver->status();
    auto t1 = std::chrono::steady_clock::now();
    const double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("probe: status() -> 0x%02x in %.3f ms\n", st, single_ms);
    std::fflush(stdout);

    // Time 100 status reads.
    constexpr int n = 100;
    t0 = std::chrono::steady_clock::now();
    uint8_t accum = 0;
    for (int i = 0; i < n; ++i) {
        accum ^= driver->status();
    }
    t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("probe: %d x status() in %.1f ms (%.3f ms/call), accum=0x%02x\n",
        n, total_ms, total_ms / n, accum);
    std::fflush(stdout);

    // One full matmul cell pattern (clear + ldw + lda*2 + seed + start +
    // status*5 + rdp = 11 calls), 100 times.
    t0 = std::chrono::steady_clock::now();
    int16_t sink = 0;
    for (int i = 0; i < n; ++i) {
        driver->clear();
        driver->ldw(0, (uint8_t) (i & 0xff));
        driver->lda(0, (int8_t) (i & 0x7f));
        driver->lda(1, (int8_t) ((i >> 1) & 0x7f));
        driver->seed(0, 0);
        driver->start();
        for (int j = 0; j < 5; ++j) (void) driver->status();
        sink ^= driver->rdp(0);
    }
    t1 = std::chrono::steady_clock::now();
    const double cell_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("probe: %d x cell (~11 calls each) in %.1f ms (%.3f ms/cell), sink=%d\n",
        n, cell_ms, cell_ms / n, (int) sink);

    // Same as above but replace the fixed 5 status() polls with the actual
    // wait_done() pattern matmul.cpp uses: poll until status & 0x02 (DONE)
    // or 128 polls. Reports the first 5 iterations' poll counts so we can
    // see if the chip reliably reports DONE in a small number of polls.
    constexpr uint8_t st_done  = 1u << 1;
    constexpr uint8_t st_error = 1u << 6;
    int max_polls_seen = 0;
    int polls_total = 0;
    int errors = 0;
    t0 = std::chrono::steady_clock::now();
    sink = 0;
    for (int i = 0; i < n; ++i) {
        driver->clear();
        driver->ldw(0, (uint8_t) (i & 0xff));
        driver->lda(0, (int8_t) (i & 0x7f));
        driver->lda(1, (int8_t) ((i >> 1) & 0x7f));
        driver->seed(0, 0);
        driver->start();
        int polls = 0;
        bool done = false;
        for (; polls < 128; ++polls) {
            const uint8_t st = driver->status();
            if (st & st_error) { errors++; break; }
            if (st & st_done)  { done = true; break; }
        }
        if (i < 5) {
            std::printf("  iter %d: polls=%d done=%d\n", i, polls, (int) done);
        }
        polls_total += polls;
        if (polls > max_polls_seen) max_polls_seen = polls;
        sink ^= driver->rdp(0);
    }
    t1 = std::chrono::steady_clock::now();
    const double wait_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("probe: %d x cell with wait_done in %.1f ms (%.3f ms/cell, mean polls=%.2f, max=%d, errors=%d), sink=%d\n",
        n, wait_ms, wait_ms / n, (double) polls_total / n, max_polls_seen, errors, (int) sink);

    return 0;
}
