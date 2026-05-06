cmake_minimum_required(VERSION 3.16)
project(bonsai_backend LANGUAGES CXX)

option(BONSAI_ENABLE_VERILATOR "Build the cycle-accurate Verilator driver as a test fixture" OFF)
option(BONSAI_BUILD_SMOKE_TESTS "Build standalone smoke-test/diagnostic executables in test/"  OFF)

# Paths into the llama.cpp checkout to build against. The build dir is expected
# to already contain libggml-base from a prior llama.cpp build (see README).
set(LLAMA_CPP_DIR        "${CMAKE_CURRENT_LIST_DIR}/llama.cpp" CACHE PATH "")
set(LLAMA_CPP_BUILD_DIR  "${LLAMA_CPP_DIR}/build-dl"           CACHE PATH "")
set(BONSAI_BACKEND_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/src/ggml-bonsai.cpp" CACHE FILEPATH "")

# Verilator inputs. RTL sources / include dirs are project-specific and must
# come from outside this CMake (see auto-discovery block below for the
# in-monorepo convenience default).
set(BONSAI_RTL_TOP          "tt_um_unclegravity_tpu" CACHE STRING "")
set(BONSAI_RTL_SOURCES      ""                       CACHE STRING "")
set(BONSAI_RTL_INCLUDE_DIRS ""                       CACHE STRING "")

set(BONSAI_INTERNAL_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/protocol.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/transport-usb.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/matmul.cpp")

find_library(GGML_BASE_LIBRARY ggml-base
    HINTS "${LLAMA_CPP_BUILD_DIR}/bin" "${LLAMA_CPP_BUILD_DIR}/lib"
    REQUIRED NO_DEFAULT_PATH)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED IMPORTED_TARGET libusb-1.0)

# Outputs land in subdirs of the build dir so production and test can share
# one cmake configuration: build/backend/libggml-bonsai.so and build/test/<exe>.
# RPATHs are relative to the binary's own directory (@loader_path / $ORIGIN),
# so each subdir is independently runnable.
set(BONSAI_BACKEND_OUTPUT_DIR "${CMAKE_BINARY_DIR}/backend")
set(BONSAI_TEST_OUTPUT_DIR    "${CMAKE_BINARY_DIR}/test")

if(APPLE)
    set(_bonsai_rpath "@loader_path;${LLAMA_CPP_BUILD_DIR}/bin;${LLAMA_CPP_BUILD_DIR}/lib")
else()
    set(_bonsai_rpath "$ORIGIN;${LLAMA_CPP_BUILD_DIR}/bin;${LLAMA_CPP_BUILD_DIR}/lib")
endif()

# Shared compile config: every target in this project links bonsai_common to
# pick up the C++ standard, ggml headers, libggml-base, and the GGML_SHARED
# define needed for ggml's dllimport/-export decorations.
add_library(bonsai_common INTERFACE)
target_compile_features(bonsai_common INTERFACE cxx_std_17)
target_include_directories(bonsai_common INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")
target_link_libraries(bonsai_common INTERFACE "${GGML_BASE_LIBRARY}" PkgConfig::LIBUSB)
target_compile_definitions(bonsai_common INTERFACE GGML_SHARED)

if(BONSAI_ENABLE_VERILATOR)
    find_package(verilator REQUIRED)

    # In-monorepo convenience: when no RTL source list is supplied and the
    # tt-tpu repo's src/rtl tree is reachable, glob it. Pass BONSAI_RTL_SOURCES
    # explicitly for any other layout (e.g. when the backend is consumed as a
    # flake input from a separate repo).
    if(NOT BONSAI_RTL_SOURCES)
        file(GLOB BONSAI_RTL_SOURCES
            "${CMAKE_CURRENT_LIST_DIR}/../../src/rtl/*.v")
        if(NOT BONSAI_RTL_SOURCES)
            message(FATAL_ERROR
                "BONSAI_ENABLE_VERILATOR=ON but BONSAI_RTL_SOURCES is empty "
                "and no defaults were found at ../../src/rtl/*.v")
        endif()
    endif()
endif()

# Verilator is a test-only fixture: the production .so does not link it.
# Test/diagnostic targets that need RTL-driven execution pull in
# test/driver-verilator.cpp via this helper.
function(bonsai_link_verilator target)
    if(NOT BONSAI_ENABLE_VERILATOR)
        return()
    endif()
    target_compile_definitions(${target} PRIVATE BONSAI_HAVE_VERILATOR=1)
    target_sources(${target} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/test/transport-verilator.cpp")
    verilate(${target}
        PREFIX Vbonsai_rtl_top
        TOP_MODULE   "${BONSAI_RTL_TOP}"
        SOURCES      ${BONSAI_RTL_SOURCES}
        INCLUDE_DIRS ${BONSAI_RTL_INCLUDE_DIRS}
        VERILATOR_ARGS --sv -Wno-fatal)
endfunction()

# The dynamic backend module loaded at runtime via GGML_BACKEND_PATH.
# Production: asic only. No verilator, no cpu fallback.
add_library(ggml-bonsai MODULE
    "${BONSAI_BACKEND_SOURCE}"
    ${BONSAI_INTERNAL_SOURCES})
target_link_libraries(ggml-bonsai PRIVATE bonsai_common)
target_compile_definitions(ggml-bonsai PRIVATE
    GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
set_target_properties(ggml-bonsai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BONSAI_BACKEND_OUTPUT_DIR}"
    BUILD_RPATH   "${_bonsai_rpath}"
    INSTALL_RPATH "${_bonsai_rpath}")
if(APPLE)
    # ggml's dynamic loader looks for *.so even on macOS.
    set_target_properties(ggml-bonsai PROPERTIES SUFFIX ".so")
endif()

if(BONSAI_BUILD_SMOKE_TESTS)
    function(bonsai_add_test name source)
        add_executable(${name} EXCLUDE_FROM_ALL ${source} ${BONSAI_INTERNAL_SOURCES})
        target_link_libraries(${name} PRIVATE bonsai_common)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${BONSAI_TEST_OUTPUT_DIR}"
            BUILD_RPATH   "${_bonsai_rpath}"
            INSTALL_RPATH "${_bonsai_rpath}")
        bonsai_link_verilator(${name})
    endfunction()

    # matmul-smoke is the verilator-driven RTL parity test; requires verilator.
    if(BONSAI_ENABLE_VERILATOR)
        bonsai_add_test(bonsai-matmul-smoke "${CMAKE_CURRENT_LIST_DIR}/test/matmul-smoke.cpp")
    endif()
    # Hardware diagnostics: bench Transport.execute round-trip, probe a real chip.
    bonsai_add_test(bonsai-bench      "${CMAKE_CURRENT_LIST_DIR}/test/bench.cpp")
    bonsai_add_test(bonsai-asic-probe "${CMAKE_CURRENT_LIST_DIR}/test/asic-probe.cpp")
    # End-to-end matmul timing: goes through matmul.cpp + USB so host-side
    # overhead (quant, dequant, plan build, fold) shows up in the numbers.
    bonsai_add_test(bonsai-matmul-bench "${CMAKE_CURRENT_LIST_DIR}/test/matmul-bench.cpp")
endif()
