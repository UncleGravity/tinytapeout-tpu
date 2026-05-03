cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

project(bonsai_backend LANGUAGES CXX)

option(BONSAI_ENABLE_VERILATOR "Build the cycle-accurate Verilator Bonsai driver" OFF)
option(BONSAI_BUILD_SMOKE_TESTS "Build standalone Bonsai backend smoke-test targets" OFF)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp"
    CACHE PATH "Path to the llama.cpp checkout to build against")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build-dl"
    CACHE PATH "Path to the llama.cpp build directory containing ggml shared libraries")
set(BONSAI_BACKEND_SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/ggml-bonsai.cpp"
    CACHE FILEPATH "Path to the Bonsai backend implementation")
set(BONSAI_RTL_TOP "tt_um_unclegravity_tpu"
    CACHE STRING "RTL top module to build with Verilator")
set(BONSAI_RTL_SOURCES ""
    CACHE STRING "Semicolon-separated RTL source files to build with Verilator")
set(BONSAI_RTL_INCLUDE_DIRS ""
    CACHE STRING "Semicolon-separated RTL include/search directories for Verilator")
set(BONSAI_VERILATOR_ARGS ""
    CACHE STRING "Additional semicolon-separated Verilator arguments")
set(BONSAI_BACKEND_INTERNAL_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/driver.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/driver-cpu.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/driver-verilator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/driver-asic.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/matmul.cpp")

set(LLAMA_CPP_GGML_INCLUDE_DIR "${LLAMA_CPP_DIR}/ggml/include")
set(LLAMA_CPP_GGML_SRC_DIR "${LLAMA_CPP_DIR}/ggml/src")
set(LLAMA_CPP_BIN_DIR "${LLAMA_CPP_BUILD_DIR}/bin")

foreach(path
        "${LLAMA_CPP_GGML_INCLUDE_DIR}/ggml.h"
        "${LLAMA_CPP_GGML_SRC_DIR}/ggml-backend-impl.h"
        "${BONSAI_BACKEND_SOURCE}"
        ${BONSAI_BACKEND_INTERNAL_SOURCES})
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "Required file not found: ${path}")
    endif()
endforeach()

find_library(GGML_BASE_LIBRARY
    NAMES ggml-base
    HINTS "${LLAMA_CPP_BIN_DIR}" "${LLAMA_CPP_BUILD_DIR}/lib" "${LLAMA_CPP_BUILD_DIR}/ggml/src"
    NO_DEFAULT_PATH)

if(NOT GGML_BASE_LIBRARY)
    message(FATAL_ERROR "Could not find libggml-base in ${LLAMA_CPP_BIN_DIR}. Build llama.cpp first.")
endif()

if(BONSAI_ENABLE_VERILATOR)
    find_package(verilator REQUIRED)

    if(NOT BONSAI_RTL_SOURCES)
        set(_bonsai_default_rtl_dir "${CMAKE_CURRENT_LIST_DIR}/../../src/rtl")
        set(_bonsai_default_rtl_sources
            "${_bonsai_default_rtl_dir}/w1a8_pe.v"
            "${_bonsai_default_rtl_dir}/w1a8_row.v"
            "${_bonsai_default_rtl_dir}/w1a8_array.v"
            "${_bonsai_default_rtl_dir}/tpu_cmd_decode.v"
            "${_bonsai_default_rtl_dir}/tpu_weight_mem.v"
            "${_bonsai_default_rtl_dir}/tpu_act_mem.v"
            "${_bonsai_default_rtl_dir}/tpu_acc_mem.v"
            "${_bonsai_default_rtl_dir}/tpu_tile_ctrl.v"
            "${_bonsai_default_rtl_dir}/tpu_status.v"
            "${_bonsai_default_rtl_dir}/project.v")

        set(_bonsai_default_rtl_available TRUE)
        foreach(path IN LISTS _bonsai_default_rtl_sources)
            if(NOT EXISTS "${path}")
                set(_bonsai_default_rtl_available FALSE)
            endif()
        endforeach()

        if(_bonsai_default_rtl_available)
            set(BONSAI_RTL_SOURCES ${_bonsai_default_rtl_sources})
        else()
            message(FATAL_ERROR
                "BONSAI_ENABLE_VERILATOR is ON, but BONSAI_RTL_SOURCES was not set "
                "and the in-repo RTL defaults were not found")
        endif()
    endif()

    foreach(path IN LISTS BONSAI_RTL_SOURCES)
        if(NOT EXISTS "${path}")
            message(FATAL_ERROR "BONSAI_RTL_SOURCES entry not found: ${path}")
        endif()
    endforeach()

    function(bonsai_add_verilator target)
        target_compile_definitions(${target}
            PRIVATE
                BONSAI_HAVE_VERILATOR=1)

        verilate(${target}
            PREFIX Vbonsai_rtl_top
            TOP_MODULE "${BONSAI_RTL_TOP}"
            SOURCES ${BONSAI_RTL_SOURCES}
            INCLUDE_DIRS ${BONSAI_RTL_INCLUDE_DIRS}
            VERILATOR_ARGS --sv -Wno-fatal ${BONSAI_VERILATOR_ARGS})
    endfunction()
endif()

add_library(ggml-bonsai MODULE
    "${BONSAI_BACKEND_SOURCE}"
    ${BONSAI_BACKEND_INTERNAL_SOURCES})

target_compile_features(ggml-bonsai PRIVATE cxx_std_17)

target_include_directories(ggml-bonsai
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include"
        "${LLAMA_CPP_GGML_INCLUDE_DIR}"
        "${LLAMA_CPP_GGML_SRC_DIR}")

target_compile_definitions(ggml-bonsai
    PRIVATE
        GGML_BACKEND_DL
        GGML_BACKEND_BUILD
        GGML_BACKEND_SHARED
        GGML_SHARED)

target_link_libraries(ggml-bonsai
    PRIVATE
        "${GGML_BASE_LIBRARY}")

if(BONSAI_ENABLE_VERILATOR)
    bonsai_add_verilator(ggml-bonsai)
endif()

set_target_properties(ggml-bonsai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(APPLE)
    set(_bonsai_rpath "@loader_path;${LLAMA_CPP_BIN_DIR};${LLAMA_CPP_BUILD_DIR}/lib")
    set_target_properties(ggml-bonsai PROPERTIES SUFFIX ".so")
else()
    set(_bonsai_rpath "$ORIGIN;${LLAMA_CPP_BIN_DIR};${LLAMA_CPP_BUILD_DIR}/lib")
endif()

set_target_properties(ggml-bonsai PROPERTIES
    BUILD_RPATH "${_bonsai_rpath}"
    INSTALL_RPATH "${_bonsai_rpath}")

if(BONSAI_BUILD_SMOKE_TESTS)
    add_executable(bonsai-matmul-smoke EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/src/matmul-smoke.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-cpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-verilator.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-asic.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/matmul.cpp")

    target_compile_features(bonsai-matmul-smoke PRIVATE cxx_std_17)

    target_include_directories(bonsai-matmul-smoke
        PRIVATE
            "${LLAMA_CPP_GGML_INCLUDE_DIR}"
            "${LLAMA_CPP_GGML_SRC_DIR}")

    target_compile_definitions(bonsai-matmul-smoke
        PRIVATE
            GGML_SHARED)

    target_link_libraries(bonsai-matmul-smoke
        PRIVATE
            "${GGML_BASE_LIBRARY}")

    set_target_properties(bonsai-matmul-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${_bonsai_rpath}"
        INSTALL_RPATH "${_bonsai_rpath}")

    if(BONSAI_ENABLE_VERILATOR)
        bonsai_add_verilator(bonsai-matmul-smoke)
    endif()

    add_executable(bonsai-driver-bench EXCLUDE_FROM_ALL
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-bench.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-cpu.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-verilator.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/driver-asic.cpp")

    target_compile_features(bonsai-driver-bench PRIVATE cxx_std_17)

    set_target_properties(bonsai-driver-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${_bonsai_rpath}"
        INSTALL_RPATH "${_bonsai_rpath}")

    if(BONSAI_ENABLE_VERILATOR)
        bonsai_add_verilator(bonsai-driver-bench)
    endif()
endif()
