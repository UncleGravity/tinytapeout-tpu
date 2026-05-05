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

    auto bench = [&](int n_tiles, int n_batches, bool one_run, const char * label) {
        bonsai::Plan plan;
        plan.ops.reserve((size_t) n_tiles);
        for (int i = 0; i < n_tiles; ++i) {
            const int8_t acts[bonsai::Tile::cols] = {
                (int8_t) (i & 0x7f),
                (int8_t) ((i >> 1) & 0x7f),
            };
            const uint8_t weights[bonsai::Tile::rows] = {
                (uint8_t) (i & 0xff),
                (uint8_t) ((i * 7) & 0xff),
            };
            const int16_t seeds[bonsai::Tile::rows] = { 0, 0 };
            const bool starts = one_run ? (i == 0)            : true;
            const bool ends   = one_run ? (i == n_tiles - 1)  : true;
            plan.add_matmul_tile_dual(weights, acts, seeds, starts, ends);
        }
        std::vector<int16_t> outs((size_t) n_tiles * (size_t) bonsai::Tile::rows);

        const auto t0 = std::chrono::steady_clock::now();
        int16_t sink = 0;
        for (int b = 0; b < n_batches; ++b) {
            transport->execute(plan, outs.data());
            for (int16_t v : outs) sink ^= v;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const int total = n_tiles * n_batches;
        std::printf("bonsai-bench: %-14s  %4d tiles/batch  %5d batches  %.1f ms total  %.3f ms/tile  (sink=%d)\n",
            label, n_tiles, n_batches, ms, ms / total, (int) sink);
    };

    // Standalone tiles (each tile is its own run — full CLEAR/SEED/RDP).
    // Single-tile Plans amortize nothing; bigger batches claw back the USB
    // round-trip cost.
    bench(1,   1024, false, "solo  1/batch");
    bench(4,    256, false, "solo  4/batch");
    bench(16,    64, false, "solo 16/batch");
    bench(64,    16, false, "solo 64/batch");
    bench(256,    4, false, "solo 256/bat");

    // Output-stationary runs. Tiles share the chip's accumulator across the
    // whole batch — only the head clears, only the tail reads back. Apples-
    // to-apples with the matching `solo N/batch` row above.
    bench(16,    64, true,  "run  16/batch");
    bench(64,    16, true,  "run  64/batch");
    bench(256,    4, true,  "run 256/batch");

    // Multi-run frames mirror what matmul.cpp now ships per cell-pair: many
    // independent Q8 sub-block runs (each 16 tiles, output-stationary)
    // packed into one X-frame so the USB round-trip is amortized across
    // the whole K dimension. K=2048 = 64 sub-blocks × 16 tiles = 1024 tiles
    // per cell-pair, executed in a single transport.execute() call.
    auto bench_multirun = [&](int run_size, int n_runs, int n_batches,
                              const char * label) {
        bonsai::Plan plan;
        const int total_tiles = run_size * n_runs;
        plan.ops.reserve((size_t) total_tiles);
        for (int r = 0; r < n_runs; ++r) {
            for (int t = 0; t < run_size; ++t) {
                const int idx = r * run_size + t;
                const int8_t acts[bonsai::Tile::cols] = {
                    (int8_t) (idx & 0x7f),
                    (int8_t) ((idx >> 1) & 0x7f),
                };
                const uint8_t weights[bonsai::Tile::rows] = {
                    (uint8_t) (idx & 0xff),
                    (uint8_t) ((idx * 7) & 0xff),
                };
                const int16_t seeds[bonsai::Tile::rows] = { 0, 0 };
                const bool starts = (t == 0);
                const bool ends   = (t == run_size - 1);
                plan.add_matmul_tile_dual(weights, acts, seeds, starts, ends);
            }
        }
        std::vector<int16_t> outs((size_t) total_tiles
                                  * (size_t) bonsai::Tile::rows);

        const auto t0 = std::chrono::steady_clock::now();
        int16_t sink = 0;
        for (int b = 0; b < n_batches; ++b) {
            transport->execute(plan, outs.data());
            for (int16_t v : outs) sink ^= v;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("bonsai-bench: %-14s  %3d runs/frame  %4d tiles/frame  %5d frames  %.1f ms total  %.3f ms/frame  (sink=%d)\n",
            label, n_runs, total_tiles, n_batches, ms, ms / n_batches, (int) sink);
    };

    // K=512 = 16 sub-blocks per cell-pair
    bench_multirun(16, 16,  64, "matmul-K512");
    // K=2048 = 64 sub-blocks per cell-pair (Bonsai-1.7B's typical hidden size)
    bench_multirun(16, 64,  16, "matmul-K2048");

    return 0;
}
