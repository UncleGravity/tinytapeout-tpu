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

// One K-tile of a matmul: `Tile::cols` activations multiplied by the
// packed-bit weight row, accumulated onto `seed`. The chip computes
// `psum = seed + Σ (a[lane] if w[lane] else -a[lane])`. The transport
// returns one int16 per op (the row-0 psum).
struct MatmulTileAttrs {
    uint8_t packed_weights;
    int8_t  acts[Tile::cols];
    int16_t seed;
};

struct PlanOp {
    OpKind kind;
    union {
        MatmulTileAttrs matmul_tile;
    } attrs;
};

struct Plan {
    std::vector<PlanOp> ops;

    void clear() { ops.clear(); }

    void add_matmul_tile(uint8_t packed_weights,
                         const int8_t * acts,
                         int16_t seed) {
        PlanOp op;
        op.kind = OpKind::MatmulTile;
        op.attrs.matmul_tile.packed_weights = packed_weights;
        for (int lane = 0; lane < Tile::cols; ++lane) {
            op.attrs.matmul_tile.acts[lane] = acts[lane];
        }
        op.attrs.matmul_tile.seed = seed;
        ops.push_back(op);
    }
};

} // namespace bonsai
