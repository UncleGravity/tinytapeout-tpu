cmake_minimum_required(VERSION 3.13)

set(PICO_PLATFORM rp2350)
set(PICO_BOARD pico2)

# pico2 sets PICO_RP2350A=1 which defaults PICO_PIO_USE_GPIO_BASE off; we need
# it on so the PIO helpers bake gpio_base=16 into pinctrl and the SM can
# address chip pins on bank 1 (uio[7]=GPIO32, uo_out=GPIO33-40).
add_compile_definitions(PICO_PIO_USE_GPIO_BASE=1)

include(pico_sdk_import.cmake)

project(rp_echo C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

add_executable(rp_echo
    main.c
    usb_descriptors.c
    ice40_loader.c
    bitstream_flash.c
    crc32.c
)

pico_generate_pio_header(rp_echo ${CMAKE_CURRENT_LIST_DIR}/chip_cycle.pio)

target_include_directories(rp_echo PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(rp_echo
    pico_stdlib
    hardware_pio
    hardware_spi
    hardware_flash
    hardware_sync
    tinyusb_device
    tinyusb_board
)

pico_add_extra_outputs(rp_echo)
