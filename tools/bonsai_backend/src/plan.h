#pragma once

#include "protocol.h"

#include <cstdint>
#include <vector>

// Typed micro-IR that flows from lowering (matmul.cpp etc.) to a Transport.
//
// Today it carries one op kind (MatmulTile)
// Future:
//   - More op kinds (SiluTile, NormPass1, …) → grow OpKind + the union.
//   - Fusion → a sequence of ops with intermediate buffers staying in
//     the chip's pipeline; today every op produces one host-side psum.
//   - On-chip memory → add `Home::Device` and a tensor-id field so a
//     weight uploaded once can be referenced by many subsequent ops.

namespace bonsai {

enum class OpKind : uint8_t {
    MatmulTile,
    // Future: SiluTile, NormPass1, NormPass2, …
};

// One K-tile of a matmul: `Tile::cols` activations multiplied by `Tile::rows`
// independent packed-bit weight rows, each accumulated onto its own seed.
// For row r the chip computes
//   psum[r] = seeds[r] + Σ (a[lane] if (packed_weights[r] >> lane) & 1
//                                  else -a[lane])
// The transport fills `Tile::rows` int16 outputs per op (one per array row).
// Callers that only need one row of work fill the unused row's weights with
// zeros; the chip still computes it (uniform tile shape on the wire) but the
// caller ignores the result.
struct MatmulTileAttrs {
    uint8_t packed_weights[Tile::rows];
    int8_t  acts[Tile::cols];
    int16_t seeds[Tile::rows];
};

// Run boundaries let consecutive MatmulTile ops share the chip's
// accumulator (output-stationary across K). Within a run only the head
// tile clears acc_q and only the tail tile reads it back; intermediate
// tiles drop CLEAR / SEED / RDP and just LDW + LDA + START. A standalone
// tile (the default) is its own run — both flags true — and emits the
// full per-tile sequence as before.
struct PlanOp {
    OpKind kind;
    bool   starts_run = true;
    bool   ends_run   = true;
    union {
        MatmulTileAttrs matmul_tile;
    } attrs;
};

struct Plan {
    std::vector<PlanOp> ops;

    void clear() { ops.clear(); }

    // Dual-row entry point: one op produces both array rows' psums in a
    // single chip fire. matmul.cpp uses this to pair adjacent output rows
    // and halve the tile count. `starts_run` / `ends_run` mark run
    // boundaries (default: standalone tile, both true). Within a multi-
    // tile run, head/tail emit CLEAR/RDP, intermediates skip them.
    void add_matmul_tile_dual(const uint8_t * packed_weights,
                              const int8_t * acts,
                              const int16_t * seeds,
                              bool starts_run = true,
                              bool ends_run = true) {
        PlanOp op;
        op.kind = OpKind::MatmulTile;
        op.starts_run = starts_run;
        op.ends_run   = ends_run;
        for (int row = 0; row < Tile::rows; ++row) {
            op.attrs.matmul_tile.packed_weights[row] = packed_weights[row];
            op.attrs.matmul_tile.seeds[row]          = seeds[row];
        }
        for (int lane = 0; lane < Tile::cols; ++lane) {
            op.attrs.matmul_tile.acts[lane] = acts[lane];
        }
        ops.push_back(op);
    }

    // Single-row convenience for tests and bench harnesses: row 1's weights
    // and seed are zeroed, and the caller is expected to read only
    // outputs[i*Tile::rows + 0]. The wire cost is identical to the dual-row
    // path because tile shape is uniform.
    void add_matmul_tile(uint8_t packed_weights,
                         const int8_t * acts,
                         int16_t seed) {
        const uint8_t weights[Tile::rows] = { packed_weights, 0 };
        int16_t       seeds  [Tile::rows] = { seed,           0 };
        add_matmul_tile_dual(weights, acts, seeds);
    }
};

} // namespace bonsai
