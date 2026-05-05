#pragma once

#include "plan.h"
#include "protocol.h"

#include <cstdint>
#include <memory>

// The seam between the lowering layer (matmul.cpp etc.) and the chip.
//
// One method does the work: execute(Plan) writes one int16 result per op
// into the caller's output buffer. A second method exposes the chip's
// status byte for end-of-matmul error checks.
//
// Two adapters live behind this seam:
//   - UsbTransport (production)         — talks to the rp_streamer firmware
//                                         over libusb bulk endpoints.
//   - VerilatorTransport (test fixture) — drives a verilated copy of the
//                                         RTL in-process, used by the
//                                         matmul-smoke parity test.

namespace bonsai {

class Transport {
public:
    virtual ~Transport() = default;

    virtual const char * name() const = 0;

    // Execute `plan` and fill `outputs[0..plan.ops.size()-1]` with one
    // int16 result per op (the row-0 psum for MatmulTile). Returns false
    // if the underlying transport died mid-execute (USB error, chip
    // dropped off the bus, etc.); on false, `outputs` is zeroed and any
    // future call will keep returning false / a dead status.
    virtual bool execute(const Plan & plan, int16_t * outputs) = 0;

    // The chip's status byte at the time of call (a Status bitfield).
    // Returns `status_error` if the transport is dead. Used by the
    // lowering layer to detect chip-side errors that accumulated across
    // an `execute` batch (the per-op error checks are dropped for
    // batching speed).
    virtual uint8_t status() = 0;
};

// Returns nullptr if no chip is found on the USB bus (libusb open failed)
// or the chip refused to come up after a USB reset.
// The backend declines to load in that case.
std::unique_ptr<Transport> create_usb_transport();

// Test fixture. Only available when the build was configured with
// -DBONSAI_ENABLE_VERILATOR=ON; otherwise the symbol is not linked and
// callers will get a link error at build time. Tests that need it live
// under tools/bonsai_backend/test/.
std::unique_ptr<Transport> create_verilator_transport();

} // namespace bonsai
