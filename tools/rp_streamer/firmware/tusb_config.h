#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

#define CFG_TUSB_OS               OPT_OS_PICO
#define CFG_TUSB_DEBUG            0

#define CFG_TUD_ENABLED           1
#define CFG_TUD_ENDPOINT0_SIZE    64

// Vendor class with raw bulk endpoints (no CDC ACM line-state framing). The
// host claims the interface via libusb and issues bulk_transfer directly,
// removing the kernel-side serial port abstraction and its small-write
// stalls.
#define CFG_TUD_CDC               0
#define CFG_TUD_VENDOR            1
#define CFG_TUD_VENDOR_RX_BUFSIZE 4096
#define CFG_TUD_VENDOR_TX_BUFSIZE 4096
#define CFG_TUD_VENDOR_EPSIZE     64

#endif
