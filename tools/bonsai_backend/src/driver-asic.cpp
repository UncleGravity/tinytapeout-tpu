#include "driver.h"

namespace bonsai {

std::unique_ptr<BonsaiDriver> create_asic_bonsai_driver() {
    // TODO: open the ASIC transport and map driver calls to the same command protocol.
    return nullptr;
}

} // namespace bonsai
