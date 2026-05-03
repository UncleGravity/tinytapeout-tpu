#ifndef _TUSB_CONFIG_H_
#define _TUSB_CONFIG_H_

#define CFG_TUSB_OS               OPT_OS_PICO
#define CFG_TUSB_DEBUG            0

#define CFG_TUD_ENABLED           1
#define CFG_TUD_ENDPOINT0_SIZE    64

#define CFG_TUD_CDC               1
#define CFG_TUD_CDC_RX_BUFSIZE    4096
#define CFG_TUD_CDC_TX_BUFSIZE    4096
#define CFG_TUD_CDC_EP_BUFSIZE    64

#endif
