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

add_library(pynq_backend_common INTERFACE)
target_compile_features(pynq_backend_common INTERFACE cxx_std_17)
target_include_directories(pynq_backend_common INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")
target_compile_definitions(pynq_backend_common INTERFACE GGML_SHARED)
target_link_libraries(pynq_backend_common INTERFACE
    "${GGML_BASE_LIBRARY}"
    nlohmann_json::nlohmann_json)

add_library(pynq_rpc STATIC "${CMAKE_CURRENT_LIST_DIR}/src/rpc.cpp")
target_link_libraries(pynq_rpc PUBLIC pynq_backend_common)
set_target_properties(pynq_rpc PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ggml-pynq MODULE "${CMAKE_CURRENT_LIST_DIR}/src/ggml-pynq.cpp")
target_link_libraries(ggml-pynq PRIVATE pynq_backend_common pynq_rpc)
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
    add_executable(pynq-backend-smoke
        "${CMAKE_CURRENT_LIST_DIR}/test/backend-smoke.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/src/ggml-pynq.cpp")
    target_link_libraries(pynq-backend-smoke PRIVATE pynq_backend_common pynq_rpc)
    set_target_properties(pynq-backend-smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PYNQ_TEST_OUTPUT_DIR}"
        BUILD_RPATH   "${_pynq_rpath}"
        INSTALL_RPATH "${_pynq_rpath}")
endif()
