cmake_minimum_required(VERSION 3.16)
project(pynq_backend LANGUAGES CXX)

option(PYNQ_BUILD_SMOKE_TESTS "Build standalone backend smoke tests" OFF)

set(LLAMA_CPP_DIR       "" CACHE PATH "Pinned llama.cpp source tree")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "Dynamic llama.cpp build/install tree")

if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "LLAMA_CPP_DIR is required")
endif()

if(NOT LLAMA_CPP_BUILD_DIR)
    message(FATAL_ERROR "LLAMA_CPP_BUILD_DIR is required")
endif()

find_library(GGML_BASE_LIBRARY ggml-base
    HINTS "${LLAMA_CPP_BUILD_DIR}/bin" "${LLAMA_CPP_BUILD_DIR}/lib"
    REQUIRED NO_DEFAULT_PATH)
find_package(nlohmann_json CONFIG REQUIRED)

set(PYNQ_BACKEND_OUTPUT_DIR "${CMAKE_BINARY_DIR}/backend")
set(PYNQ_TEST_OUTPUT_DIR    "${CMAKE_BINARY_DIR}/test")

if(APPLE)
    set(_pynq_rpath "@loader_path;${LLAMA_CPP_BUILD_DIR}/bin;${LLAMA_CPP_BUILD_DIR}/lib")
else()
    set(_pynq_rpath "$ORIGIN;${LLAMA_CPP_BUILD_DIR}/bin;${LLAMA_CPP_BUILD_DIR}/lib")
endif()

set(PYNQ_TRANSPORT_DIR  "${CMAKE_CURRENT_LIST_DIR}/../transport")
set(PYNQ_PROFILING_DIR  "${CMAKE_CURRENT_LIST_DIR}/../profiling")
set(PYNQ_PROTO_DIR      "${CMAKE_CURRENT_LIST_DIR}/../../proto")
set(PYNQ_E2E_TESTS_DIR  "${CMAKE_CURRENT_LIST_DIR}/../../tests/e2e")

add_library(pynq_backend_common INTERFACE)
target_compile_features(pynq_backend_common INTERFACE cxx_std_17)
target_include_directories(pynq_backend_common INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${PYNQ_TRANSPORT_DIR}"
    "${PYNQ_PROFILING_DIR}"
    "${PYNQ_PROTO_DIR}/.."
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")
target_compile_definitions(pynq_backend_common INTERFACE GGML_SHARED)
target_link_libraries(pynq_backend_common INTERFACE
    "${GGML_BASE_LIBRARY}"
    nlohmann_json::nlohmann_json)

add_library(pynq_profiling STATIC "${PYNQ_PROFILING_DIR}/events.cpp")
target_link_libraries(pynq_profiling PUBLIC pynq_backend_common)
set_target_properties(pynq_profiling PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pynq_transport STATIC "${PYNQ_TRANSPORT_DIR}/client.cpp")
target_link_libraries(pynq_transport PUBLIC pynq_backend_common pynq_profiling)
set_target_properties(pynq_transport PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(PYNQ_BACKEND_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/src/device.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/buffer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/lowering.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/trace.cpp")

add_library(ggml-pynq MODULE ${PYNQ_BACKEND_SOURCES})
target_link_libraries(ggml-pynq PRIVATE pynq_backend_common pynq_transport)
target_compile_definitions(ggml-pynq PRIVATE
    GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
set_target_properties(ggml-pynq PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYNQ_BACKEND_OUTPUT_DIR}"
    BUILD_RPATH   "${_pynq_rpath}"
    INSTALL_RPATH "${_pynq_rpath}")
if(APPLE)
    # ggml's dynamic backend loader looks for *.so on Darwin too.
    set_target_properties(ggml-pynq PROPERTIES SUFFIX ".so")
endif()

if(PYNQ_BUILD_SMOKE_TESTS)
    find_library(GGML_LIBRARY ggml
        HINTS "${LLAMA_CPP_BUILD_DIR}/lib"
        REQUIRED NO_DEFAULT_PATH)
    set(PYNQ_SMOKE_TEST_DIR "${PYNQ_E2E_TESTS_DIR}/backend")
    set(PYNQ_SMOKE_TEST_SOURCES
        "${PYNQ_SMOKE_TEST_DIR}/main.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/common.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_upload_download.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_copy.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_matmul.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_swiglu.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_glue.cpp"
        "${PYNQ_SMOKE_TEST_DIR}/test_multi_extent.cpp")
    add_executable(pynq-backend-smoke
        ${PYNQ_SMOKE_TEST_SOURCES}
        ${PYNQ_BACKEND_SOURCES})
    target_link_libraries(pynq-backend-smoke PRIVATE
        pynq_backend_common
        pynq_transport
        "${GGML_LIBRARY}")
    target_compile_definitions(pynq-backend-smoke PRIVATE
        PYNQ_GGML_BACKEND_DIR="${LLAMA_CPP_BUILD_DIR}/bin")
    set_target_properties(pynq-backend-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PYNQ_TEST_OUTPUT_DIR}"
        BUILD_RPATH   "${_pynq_rpath}"
        INSTALL_RPATH "${_pynq_rpath}")
endif()
