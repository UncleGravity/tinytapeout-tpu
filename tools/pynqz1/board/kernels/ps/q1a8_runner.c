/*
 * q1a8_runner.c — single-call C orchestration of one Q1A8 matmul chunk.
 *
 * Replaces ~10 Python sections (kernel_setup, recv_start, kernel_start,
 * acts_send_start, weights_send_start, *_wait, poll, ...) with one C call
 * that pokes AXI DMA + kernel registers directly. Removes the per-section
 * ctypes/PYNQ-wrapper Python overhead which was ~8ms/matmul on Cortex-A9.
 *
 * The caller (PLMatmulQ1A8.run) is responsible for:
 *   - cache flush/invalidate on the acts and result buffers
 *   - resolving the weights buffer's physical address (slab pointer or
 *     multi-extent scratch)
 *   - quantizing + packing acts before the call
 *
 * AXI DMA register layout follows Xilinx PG021 (Direct Register Mode):
 *   0x00 MM2S_DMACR    bit0=RS, bit2=Reset
 *   0x04 MM2S_DMASR    bit0=Halted, bit1=Idle, bits[6:4]=err
 *   0x18 MM2S_SA       source address
 *   0x28 MM2S_LENGTH   length in bytes (writing triggers transfer)
 *   0x30 S2MM_DMACR    same layout as MM2S
 *   0x34 S2MM_DMASR
 *   0x48 S2MM_DA       destination address
 *   0x58 S2MM_LENGTH
 */

#include <stdint.h>
#include <stddef.h>

/* AXI DMA register offsets, in uint32_t units (word index). */
#define DMA_MM2S_DMACR    (0x00 / 4)
#define DMA_MM2S_DMASR    (0x04 / 4)
#define DMA_MM2S_SA       (0x18 / 4)
#define DMA_MM2S_LENGTH   (0x28 / 4)
#define DMA_S2MM_DMACR    (0x30 / 4)
#define DMA_S2MM_DMASR    (0x34 / 4)
#define DMA_S2MM_DA       (0x48 / 4)
#define DMA_S2MM_LENGTH   (0x58 / 4)

#define DMACR_RS          (1u << 0)
#define DMACR_RESET       (1u << 2)
#define DMASR_HALTED      (1u << 0)
#define DMASR_IDLE        (1u << 1)
#define DMASR_ERR_MASK    (0x70u)  /* DMAIntErr | DMASlvErr | DMADecErr */

/* q1a8_kernel_top register offsets, in uint32_t units. */
#define KREG_CTRL           (0x08 / 4)
#define KREG_STATUS         (0x0C / 4)
#define KREG_NUM_Q1_BLOCKS  (0x10 / 4)
#define KREG_NUM_ROWBLOCKS  (0x14 / 4)
#define KREG_CYCLES         (0x18 / 4)

#define KCTRL_START         (1u << 0)
#define KSTAT_DONE          (1u << 1)

static inline uint32_t mmio_rd(volatile uint32_t * base, int off) {
    return base[off];
}
static inline void mmio_wr(volatile uint32_t * base, int off, uint32_t val) {
    base[off] = val;
}

/* Bring one DMA channel out of reset and into running mode if it isn't
 * already. PYNQ's first transfer() call would do this in Python; doing
 * it here means C-only orchestration after construction. */
static int ensure_dma_running(volatile uint32_t * dma, int dmacr_off) {
    const uint32_t cr = mmio_rd(dma, dmacr_off);
    if ((cr & DMACR_RS) != 0) {
        return 0;
    }
    mmio_wr(dma, dmacr_off, DMACR_RESET);
    for (uint32_t i = 0; i < 1000000u; ++i) {
        if ((mmio_rd(dma, dmacr_off) & DMACR_RESET) == 0) {
            break;
        }
    }
    if (mmio_rd(dma, dmacr_off) & DMACR_RESET) {
        return -1;
    }
    mmio_wr(dma, dmacr_off, DMACR_RS);
    return 0;
}

/* Poll DMASR until Idle=1 or an error bit fires. Returns 0 on idle,
 * -1 on timeout, -2 on DMA error. */
static int poll_dma_idle(volatile uint32_t * dma, int dmasr_off,
                         uint32_t poll_limit) {
    for (uint32_t i = 0; i < poll_limit; ++i) {
        const uint32_t sr = mmio_rd(dma, dmasr_off);
        if (sr & DMASR_ERR_MASK) {
            return -2;
        }
        if (sr & DMASR_IDLE) {
            return 0;
        }
    }
    return -1;
}

/*
 * One Q1A8 matmul chunk: configure kernel, kick all three DMAs, busy-poll
 * for completion, return cycle count.
 *
 * dma_w_regs is axi_dma_0 (carries weights MM2S + result S2MM).
 * dma_a_regs is axi_dma_1 (acts MM2S only).
 *
 * poll_limit is per-poll-loop iteration cap. ~100k is enough for any
 * matmul we ship today; larger values are safe but waste cycles on
 * genuine errors before returning -1.
 *
 * Error codes (negative): -1 generic arg, -10..-12 DMA reset failures,
 * -20/-21 acts MM2S timeout/err, -30/-31 weights MM2S, -40/-41 S2MM,
 * -50 kernel DONE timeout. Caller maps these to a RuntimeError.
 */
int bonsai_q1a8_run_matmul_chunk(
    volatile uint32_t * kernel_regs,
    volatile uint32_t * dma_w_regs,
    volatile uint32_t * dma_a_regs,
    uint32_t weights_phys_addr,
    uint32_t weights_nbytes,
    uint32_t acts_phys_addr,
    uint32_t acts_nbytes,
    uint32_t result_phys_addr,
    uint32_t result_nbytes,
    uint32_t num_q1_blocks,
    uint32_t num_rowblocks,
    uint32_t poll_limit,
    uint32_t * out_cycles
) {
    if (kernel_regs == NULL || dma_w_regs == NULL || dma_a_regs == NULL ||
        out_cycles == NULL) {
        return -1;
    }

    if (ensure_dma_running(dma_w_regs, DMA_MM2S_DMACR) != 0) return -10;
    if (ensure_dma_running(dma_w_regs, DMA_S2MM_DMACR) != 0) return -11;
    if (ensure_dma_running(dma_a_regs, DMA_MM2S_DMACR) != 0) return -12;

    /* Configure kernel for this chunk. */
    mmio_wr(kernel_regs, KREG_NUM_Q1_BLOCKS, num_q1_blocks);
    mmio_wr(kernel_regs, KREG_NUM_ROWBLOCKS, num_rowblocks);

    /* Arm S2MM (result sink) BEFORE starting the kernel — the kernel begins
     * emitting after the first rowblock; the sink must be ready. */
    mmio_wr(dma_w_regs, DMA_S2MM_DA, result_phys_addr);
    mmio_wr(dma_w_regs, DMA_S2MM_LENGTH, result_nbytes);

    /* Start kernel. */
    mmio_wr(kernel_regs, KREG_CTRL, KCTRL_START);

    /* Start MM2S for both streams. The kernel's LOAD_ACTS state consumes
     * the acts stream first; weights are back-pressured by the kernel's
     * tready until then. Issuing both in close succession lets the AXI
     * interconnect overlap them. */
    mmio_wr(dma_a_regs, DMA_MM2S_SA, acts_phys_addr);
    mmio_wr(dma_a_regs, DMA_MM2S_LENGTH, acts_nbytes);

    mmio_wr(dma_w_regs, DMA_MM2S_SA, weights_phys_addr);
    mmio_wr(dma_w_regs, DMA_MM2S_LENGTH, weights_nbytes);

    /* Wait for both sends + result recv. Order doesn't matter for
     * correctness, only for the worst-case wall time of this function. */
    int rc;
    rc = poll_dma_idle(dma_a_regs, DMA_MM2S_DMASR, poll_limit);
    if (rc != 0) return -20 + rc;  /* -21 or -22 */

    rc = poll_dma_idle(dma_w_regs, DMA_MM2S_DMASR, poll_limit);
    if (rc != 0) return -30 + rc;

    rc = poll_dma_idle(dma_w_regs, DMA_S2MM_DMASR, poll_limit);
    if (rc != 0) return -40 + rc;

    /* Final fence: kernel STATUS.done. By the time S2MM is idle the kernel
     * has emitted everything but it sets done one cycle later. */
    for (uint32_t i = 0; i < poll_limit; ++i) {
        if (mmio_rd(kernel_regs, KREG_STATUS) & KSTAT_DONE) {
            *out_cycles = mmio_rd(kernel_regs, KREG_CYCLES);
            return 0;
        }
    }
    return -50;
}
