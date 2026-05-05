#include "plan.h"
#include "protocol.h"
#include "transport.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

// Hardware bring-up probe: opens the USB transport and reports per-call
// timings for status() and a couple of Plan shapes. Useful for sanity-
// checking that a real chip is alive and responsive after firmware/RTL
// changes. Distinct from bench.cpp in that it prints individual numbers
// rather than a sweep table.

int main() {
    setvbuf(stderr, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("probe: opening USB transport...\n");
    auto transport = bonsai::create_usb_transport();
    if (transport == nullptr) {
        std::fprintf(stderr, "probe: create_usb_transport returned null\n");
        return 1;
    }
    std::printf("probe: transport=%s open\n", transport->name());
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;

    // Single status() round-trip.
    {
        auto t0 = clock::now();
        const uint8_t st = transport->status();
        auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("probe: status() -> 0x%02x in %.3f ms\n", st, ms);
    }

    // 100 status() reads in a row.
    {
        constexpr int n = 100;
        auto t0 = clock::now();
        uint8_t accum = 0;
        for (int i = 0; i < n; ++i) accum ^= transport->status();
        auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("probe: %d x status() in %.1f ms (%.3f ms/call), accum=0x%02x\n",
            n, ms, ms / n, accum);
    }

    // 100 single-tile Plans — measures per-X-frame round-trip.
    {
        constexpr int n = 100;
        bonsai::Plan plan;
        std::vector<int16_t> outs(1);
        auto t0 = clock::now();
        int16_t sink = 0;
        for (int i = 0; i < n; ++i) {
            const int8_t acts[bonsai::Tile::cols] = {
                (int8_t) (i & 0x7f),
                (int8_t) ((i >> 1) & 0x7f),
            };
            plan.clear();
            plan.add_matmul_tile((uint8_t) (i & 0xff), acts, /*seed=*/ 0);
            transport->execute(plan, outs.data());
            sink ^= outs[0];
        }
        auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("probe: %d x 1-tile Plan in %.1f ms (%.3f ms/tile), sink=%d\n",
            n, ms, ms / n, (int) sink);
    }

    // Batched Plans at varying sizes — show how much per-tile cost we
    // claw back by amortizing the USB round-trip.
    for (int batch_size : { 1, 4, 16, 64, 256 }) {
        const int total_tiles = 1024;
        const int n_batches = total_tiles / batch_size;
        bonsai::Plan plan;
        plan.ops.reserve((size_t) batch_size);
        std::vector<int16_t> outs((size_t) batch_size);
        auto t0 = clock::now();
        int16_t sink = 0;
        for (int b = 0; b < n_batches; ++b) {
            plan.clear();
            for (int i = 0; i < batch_size; ++i) {
                const int idx = b * batch_size + i;
                const int8_t acts[bonsai::Tile::cols] = {
                    (int8_t) (idx & 0x7f),
                    (int8_t) ((idx >> 1) & 0x7f),
                };
                plan.add_matmul_tile((uint8_t) (idx & 0xff), acts, /*seed=*/ 0);
            }
            transport->execute(plan, outs.data());
            for (int16_t v : outs) sink ^= v;
        }
        auto t1 = clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("probe: batch=%-3d  %d tiles in %.1f ms (%.3f ms/tile), sink=%d\n",
            batch_size, total_tiles, ms, ms / total_tiles, (int) sink);
    }

    return 0;
}
