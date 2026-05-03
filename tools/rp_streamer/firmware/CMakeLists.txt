cmake_minimum_required(VERSION 3.13)

set(PICO_PLATFORM rp2350)
set(PICO_BOARD pico2)

include(pico_sdk_import.cmake)

project(rp_echo C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

add_executable(rp_echo
    main.c
    usb_descriptors.c
)

target_include_directories(rp_echo PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(rp_echo
    pico_stdlib
    tinyusb_device
    tinyusb_board
)

pico_add_extra_outputs(rp_echo)
