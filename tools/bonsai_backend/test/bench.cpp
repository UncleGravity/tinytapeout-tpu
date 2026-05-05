#include "plan.h"
#include "protocol.h"
#include "transport.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

// Bench end-to-end Transport.execute(Plan) latency. Hardware-required;
// uses the USB transport so the numbers reflect what real inference
// actually pays per matmul tile.

int main() {
    auto transport = bonsai::create_usb_transport();
    if (transport == nullptr) {
        std::fprintf(stderr, "bonsai-bench: USB transport unavailable (no chip on bus?)\n");
        return 1;
    }

    auto bench = [&](int n_tiles, int n_batches, const char * label) {
        bonsai::Plan plan;
        plan.ops.reserve((size_t) n_tiles);
        for (int i = 0; i < n_tiles; ++i) {
            const int8_t acts[bonsai::Tile::cols] = {
                (int8_t) (i & 0x7f),
                (int8_t) ((i >> 1) & 0x7f),
            };
            plan.add_matmul_tile((uint8_t) (i & 0xff), acts, /*seed=*/ 0);
        }
        std::vector<int16_t> outs((size_t) n_tiles);

        const auto t0 = std::chrono::steady_clock::now();
        int16_t sink = 0;
        for (int b = 0; b < n_batches; ++b) {
            transport->execute(plan, outs.data());
            for (int16_t v : outs) sink ^= v;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int total = n_tiles * n_batches;
        std::printf("bonsai-bench: %-12s  %4d tiles/batch  %5d batches  %.1f ms total  %.3f ms/tile  (sink=%d)\n",
            label, n_tiles, n_batches, ms, ms / total, (int) sink);
    };

    // Single-tile Plans amortize nothing — measures per-X-frame round-trip.
    bench(1,   1024, "1/batch");
    // Increasingly batched Plans — show how much per-tile cost we claw back
    // by amortizing the USB round-trip.
    bench(4,    256, "4/batch");
    bench(16,    64, "16/batch");
    bench(64,    16, "64/batch");
    bench(256,    4, "256/batch");

    return 0;
}
