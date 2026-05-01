cmake_minimum_required(VERSION 3.16)

project(bonsai_backend LANGUAGES CXX)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/llama.cpp"
    CACHE PATH "Path to the llama.cpp checkout to build against")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build-dl"
    CACHE PATH "Path to the llama.cpp build directory containing ggml shared libraries")
set(BONSAI_BACKEND_SOURCE "${CMAKE_CURRENT_LIST_DIR}/src/ggml-bonsai.cpp"
    CACHE FILEPATH "Path to the Bonsai backend implementation")

set(LLAMA_CPP_GGML_INCLUDE_DIR "${LLAMA_CPP_DIR}/ggml/include")
set(LLAMA_CPP_GGML_SRC_DIR "${LLAMA_CPP_DIR}/ggml/src")
set(LLAMA_CPP_BIN_DIR "${LLAMA_CPP_BUILD_DIR}/bin")

foreach(path
        "${LLAMA_CPP_GGML_INCLUDE_DIR}/ggml.h"
        "${LLAMA_CPP_GGML_SRC_DIR}/ggml-backend-impl.h"
        "${BONSAI_BACKEND_SOURCE}")
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

add_library(ggml-bonsai MODULE "${BONSAI_BACKEND_SOURCE}")

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
