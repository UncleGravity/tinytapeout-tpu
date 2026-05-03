#include "driver.h"

#include <cstdlib>
#include <cstring>

namespace bonsai {

const char * driver_kind_name(DriverKind kind) {
    switch (kind) {
        case DriverKind::cpu:
            return "cpu";
        case DriverKind::verilator:
            return "verilator";
        case DriverKind::asic:
            return "asic";
    }

    return "unknown";
}

DriverKind driver_kind_from_env() {
    const char * value = std::getenv("BONSAI_DRIVER");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "cpu") == 0) {
        return DriverKind::cpu;
    }
    if (std::strcmp(value, "verilator") == 0) {
        return DriverKind::verilator;
    }
    if (std::strcmp(value, "asic") == 0) {
        return DriverKind::asic;
    }

    return DriverKind::cpu;
}

std::unique_ptr<BonsaiDriver> create_bonsai_driver(DriverKind kind) {
    switch (kind) {
        case DriverKind::cpu:
            return create_cpu_bonsai_driver();
        case DriverKind::verilator:
            return create_verilator_bonsai_driver();
        case DriverKind::asic:
            return create_asic_bonsai_driver();
    }

    return nullptr;
}

} // namespace bonsai
