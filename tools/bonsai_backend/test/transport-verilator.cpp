#include "transport.h"
#include "protocol.h"

#if defined(BONSAI_HAVE_VERILATOR)
#include "Vbonsai_rtl_top.h"

#include <verilated.h>
#endif

#include <cstdint>
#include <cstring>
#include <memory>

#if defined(BONSAI_HAVE_VERILATOR)
double sc_time_stamp() {
    return 0.0;
}
#endif

// Test fixture: drives a verilated copy of the RTL in-process. Used by
// bonsai-matmul-smoke as a parity test for the lowering layer (matmul.cpp
// + protocol.cpp + the RTL together).
//
// One instance owns one VerilatedContext + Vbonsai_rtl_top — independent
// state per transport, so cloning would be cheap (we don't, today).

namespace bonsai {

#if defined(BONSAI_HAVE_VERILATOR)

namespace {

class VerilatorTransport final : public Transport {
public:
    VerilatorTransport() :
        context(new VerilatedContext),
        top(new Vbonsai_rtl_top(context.get(), "bonsai_rtl_top")) {
        reset();
    }

    ~VerilatorTransport() override {
        top->final();
    }

    const char * name() const override { return "verilator"; }

    bool execute(const Plan & plan, int16_t * outputs) override {
        for (size_t i = 0; i < plan.ops.size(); ++i) {
            const PlanOp & op = plan.ops[i];
            // Today only MatmulTile exists; future op kinds will dispatch
            // here too.
            const MatmulTileAttrs & m = op.attrs.matmul_tile;
            outputs[i] = run_one_matmul_tile(m);
        }
        return true;
    }

    uint8_t status() override {
        return transact(cmd_status);
    }

private:
    std::unique_ptr<VerilatedContext> context;
    std::unique_ptr<Vbonsai_rtl_top> top;

    void reset() {
        top->ena = 1;
        top->ui_in = 0;
        top->uio_in = 0;
        top->clk = 0;
        top->rst_n = 0;
        top->eval();

        for (int i = 0; i < 4; ++i) tick();

        top->rst_n = 1;
        top->eval();

        for (int i = 0; i < 2; ++i) tick();
    }

    void tick() {
        top->clk = 1;
        top->eval();
        context->timeInc(1);

        top->clk = 0;
        top->eval();
        context->timeInc(1);
    }

    // One chip-cycle: drive ui_in/uio_in, sample uo_out at the rising edge.
    uint8_t transact(Command cmd, uint8_t data = 0, uint8_t arg = 0) {
        top->ui_in = pack_ui(cmd, arg);
        top->uio_in = data;
        top->eval();

        top->clk = 1;
        top->eval();
        const uint8_t out = (uint8_t) top->uo_out;
        context->timeInc(1);

        top->clk = 0;
        top->eval();
        context->timeInc(1);

        return out;
    }

    // Inline equivalent of the chip's per-tile sequence. Mirrors what
    // protocol::build_tile_pairs packs for the USB transport — the two
    // paths must produce the same result for any given tile, which is
    // exactly what bonsai-matmul-smoke verifies.
    int16_t run_one_matmul_tile(const MatmulTileAttrs & m) {
        transact(cmd_clear);
        transact(cmd_nop);
        transact(cmd_ldw, m.packed_weights, encode_row_arg(0));
        for (int lane = 0; lane < Tile::cols; ++lane) {
            transact(cmd_lda, (uint8_t) m.acts[lane], encode_col_arg(lane));
        }
        const uint16_t seed_raw = (uint16_t) m.seed;
        for (int byte = 0; byte < psum_bytes; ++byte) {
            transact(cmd_seed,
                     (uint8_t) ((seed_raw >> (byte * 8)) & 0xffu),
                     encode_row_byte_arg(0, byte));
        }
        transact(cmd_start);
        transact(cmd_nop);

        // Wait for DONE. The chip's tile_done_pad_cycles bound is enough
        // for the X-frame path; here we poll defensively up to 128 cycles.
        constexpr int max_polls = 128;
        for (int i = 0; i < max_polls; ++i) {
            const uint8_t st = transact(cmd_status);
            if (st & status_error) return 0;
            if (st & status_done)  break;
        }

        uint16_t raw = 0;
        for (int byte = 0; byte < psum_bytes; ++byte) {
            raw |= (uint16_t) transact(cmd_rdp, 0, encode_row_byte_arg(0, byte))
                   << (byte * 8);
        }
        return sign_extend_psum(raw);
    }
};

} // namespace

std::unique_ptr<Transport> create_verilator_transport() {
    return std::unique_ptr<Transport>(new VerilatorTransport());
}

#endif // BONSAI_HAVE_VERILATOR

} // namespace bonsai
