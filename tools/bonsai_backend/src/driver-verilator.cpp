#include "driver.h"

#if defined(BONSAI_HAVE_VERILATOR)
#include "Vbonsai_rtl_top.h"

#include <verilated.h>
#endif

#if defined(BONSAI_HAVE_VERILATOR)
double sc_time_stamp() {
    return 0.0;
}
#endif

namespace bonsai {

namespace {

#if defined(BONSAI_HAVE_VERILATOR)

enum Command : uint8_t {
    cmd_status = 0,
    cmd_clear  = 1,
    cmd_ldw    = 2,
    cmd_lda    = 3,
    cmd_seed   = 4,
    cmd_start  = 5,
    cmd_rdp    = 6,
    cmd_nop    = 7,
};

constexpr int bit_width_for_count(int count) {
    int bits = 1;
    int max_value = count - 1;
    while ((1 << bits) <= max_value) {
        ++bits;
    }
    return bits;
}

constexpr int row_bits = bit_width_for_count(Tile::rows);
constexpr int col_bits = bit_width_for_count(Tile::cols);
constexpr int psum_bytes = (Tile::psum_bits + 7) / 8;

static_assert(Tile::cols <= 8, "LDW packs one tile row into an 8-bit data byte");

uint8_t pack_ui(Command cmd, uint8_t arg = 0) {
    return (uint8_t) cmd | (uint8_t) ((arg & 0x1fu) << 3);
}

uint8_t encode_row_arg(int row) {
    return (uint8_t) (row & ((1 << row_bits) - 1));
}

uint8_t encode_col_arg(int col) {
    return (uint8_t) (col & ((1 << col_bits) - 1));
}

uint8_t encode_row_byte_arg(int row, int byte) {
    return (uint8_t) (encode_row_arg(row) | (byte << row_bits));
}

int16_t sign_extend_psum(uint16_t raw) {
    constexpr uint16_t sign = 1u << (Tile::psum_bits - 1);
    constexpr uint16_t mask = (Tile::psum_bits == 16) ? 0xffffu : ((1u << Tile::psum_bits) - 1u);
    raw &= mask;
    return (raw & sign) ? (int16_t) ((int32_t) raw - (int32_t) (mask + 1u)) : (int16_t) raw;
}

class VerilatorBonsaiDriver final : public BonsaiDriver {
public:
    VerilatorBonsaiDriver() :
        context(new VerilatedContext),
        top(new Vbonsai_rtl_top(context.get(), "bonsai_rtl_top")) {
        reset();
    }

    ~VerilatorBonsaiDriver() override {
        top->final();
    }

    const char * name() const override {
        return "verilator";
    }

    void clear() override {
        transact(cmd_clear);
        transact(cmd_nop);
    }

    void ldw(int row, uint8_t packed_weights) override {
        transact(cmd_ldw, packed_weights, encode_row_arg(row));
    }

    void lda(int col, int8_t act) override {
        transact(cmd_lda, (uint8_t) act, encode_col_arg(col));
    }

    void seed(int row, int16_t psum) override {
        const uint16_t raw = (uint16_t) psum;
        for (int byte = 0; byte < psum_bytes; ++byte) {
            transact(
                cmd_seed,
                (uint8_t) ((raw >> (byte * 8)) & 0xffu),
                encode_row_byte_arg(row, byte));
        }
    }

    void start() override {
        transact(cmd_start);
        transact(cmd_nop);
    }

    uint8_t status() override {
        return transact(cmd_status);
    }

    int16_t rdp(int row) override {
        uint16_t raw = 0;
        for (int byte = 0; byte < psum_bytes; ++byte) {
            raw |= (uint16_t) transact(cmd_rdp, 0, encode_row_byte_arg(row, byte)) << (byte * 8);
        }
        return sign_extend_psum(raw);
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

        for (int i = 0; i < 4; ++i) {
            tick();
        }

        top->rst_n = 1;
        top->eval();

        for (int i = 0; i < 2; ++i) {
            tick();
        }
    }

    void tick() {
        top->clk = 1;
        top->eval();
        context->timeInc(1);

        top->clk = 0;
        top->eval();
        context->timeInc(1);
    }

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
};

#endif

} // namespace

std::unique_ptr<BonsaiDriver> create_verilator_bonsai_driver() {
#if defined(BONSAI_HAVE_VERILATOR)
    return std::unique_ptr<BonsaiDriver>(new VerilatorBonsaiDriver());
#else
    return nullptr;
#endif
}

} // namespace bonsai
