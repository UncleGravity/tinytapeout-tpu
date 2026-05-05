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
            int16_t * psum_slot = &outputs[i * (size_t) Tile::rows];
            run_one_matmul_tile(m, op.starts_run, op.ends_run, psum_slot);
            if (!op.ends_run) {
                // Mid-run tiles don't have psums to read yet — match the
                // USB transport's behavior of leaving zeros in their slot.
                for (int row = 0; row < Tile::rows; ++row) {
                    psum_slot[row] = 0;
                }
            }
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
    // exactly what bonsai-matmul-smoke verifies. When `starts_run` is
    // false, CLEAR/SEED are skipped (the chip's acc_q persists from the
    // previous tile in the run). When `ends_run` is false, RDP is skipped
    // and `out_psums` is left untouched by this call — the run's tail
    // tile reads them out for everyone.
    void run_one_matmul_tile(const MatmulTileAttrs & m,
                             bool starts_run,
                             bool ends_run,
                             int16_t * out_psums) {
        if (starts_run) {
            transact(cmd_clear);
            transact(cmd_nop);
        }
        for (int row = 0; row < Tile::rows; ++row) {
            transact(cmd_ldw, m.packed_weights[row], encode_row_arg(row));
        }
        for (int lane = 0; lane < Tile::cols; ++lane) {
            transact(cmd_lda, (uint8_t) m.acts[lane], encode_col_arg(lane));
        }
        if (starts_run) {
            for (int row = 0; row < Tile::rows; ++row) {
                const uint16_t seed_raw = (uint16_t) m.seeds[row];
                for (int byte = 0; byte < psum_bytes; ++byte) {
                    transact(cmd_seed,
                             (uint8_t) ((seed_raw >> (byte * 8)) & 0xffu),
                             encode_row_byte_arg(row, byte));
                }
            }
        }
        transact(cmd_start);
        transact(cmd_nop);

        // Wait for DONE. The chip's tile_done_pad_cycles bound is enough
        // for the X-frame path; here we poll defensively up to 128 cycles.
        constexpr int max_polls = 128;
        bool errored = false;
        for (int i = 0; i < max_polls; ++i) {
            const uint8_t st = transact(cmd_status);
            if (st & status_error) { errored = true; break; }
            if (st & status_done)  break;
        }

        if (!ends_run) {
            // No RDP this tile — the run-tail tile reads the final acc_q.
            return;
        }
        for (int row = 0; row < Tile::rows; ++row) {
            if (errored) {
                out_psums[row] = 0;
                continue;
            }
            uint16_t raw = 0;
            for (int byte = 0; byte < psum_bytes; ++byte) {
                raw |= (uint16_t) transact(cmd_rdp, 0,
                                           encode_row_byte_arg(row, byte))
                       << (byte * 8);
            }
            out_psums[row] = sign_extend_psum(raw);
        }
    }
};

} // namespace

std::unique_ptr<Transport> create_verilator_transport() {
    return std::unique_ptr<Transport>(new VerilatorTransport());
}

#endif // BONSAI_HAVE_VERILATOR

} // namespace bonsai
