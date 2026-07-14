#ifndef CAPSTONE_TESTS_RTL_SMOKE_FPGA_INSTRUMENT_H
#define CAPSTONE_TESTS_RTL_SMOKE_FPGA_INSTRUMENT_H

/* BORROW_COST_SLOT_* / BORROW_COST_REGION_SIZE live in the shared probe header. */
#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"

/* Hardware instrumentation shim for the RTL/FPGA borrow-cost port (task-016).
 *
 * The QEMU probe (tests/runtime-qemu/borrow-cost-probe/) measures with two
 * emulator-only custom ops that DO NOT EXIST on the CVA6/Capstone silicon:
 *   - csrdicount  (.insn r 0x5b,0x1,0x48)  -> QEMU retired-instruction count
 *   - csdebugcount* (0x5b,0x1,0x45..0x48)  -> QEMU serial-log counters
 * On the FPGA we replace them with real hardware:
 *   - counting: the standard RISC-V cycle counter (rdcycle / mcycle CSR);
 *   - output:   the domain writes results into the shared region and the
 *               controller (ordinary Linux userspace) printf()s them over UART.
 *
 * This header is included ONLY by the FPGA domain payload (borrow_cost_fpga.c).
 * It carries no capability builtins, so it is safe next to the shared header.
 */

/* --- cycle read -----------------------------------------------------------
 *
 * `rdcycle` reads the unprivileged `cycle` CSR (0xC00), a mirror of the M-mode
 * `mcycle`. It is readable from S/U mode ONLY if the higher privilege level
 * enables it: mcounteren.CY (M->S) and scounteren.CY (S->U). Linux normally
 * sets scounteren.CY, so a Linux userspace process can rdcycle; whether a
 * *Capstone domain* may is set by the monitor's counteren for domains.
 *
 * OPEN ITEM (verify on first boot): if `rdcycle` faults inside the domain, the
 * monitor is not exposing the counter to domains. Two fixes, in order of
 * preference:
 *   (a) have the monitor set [m|s]counteren.CY for the domain context, or
 *   (b) run the measurement bare-metal in M-mode and read `mcycle` directly
 *       (csrr %0, mcycle) -- mcycle is always M-mode readable (confirmed in
 *       capstone-ariane core/csr_regfile.sv:677).
 * We default to (a)+rdcycle because it keeps the existing controller+domain
 * model (this file) unchanged; (b) is the fallback and only swaps the asm below.
 */
static inline unsigned long rd_cycles(void) {
  unsigned long v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}

/* The QEMU probe's measured loops call rd_icount(); alias it to the hardware
 * cycle read so those loop bodies stay byte-identical to the QEMU version. */
static inline unsigned long rd_icount(void) { return rd_cycles(); }

/* --- result hand-off ------------------------------------------------------
 *
 * The domain writes eight unsigned-long results to the base of the shared
 * region, at the slot indices already defined in borrow_cost_probe.h
 * (BORROW_COST_SLOT_*). The controller maps the same region and reads them
 * back. 8 * 8 = 64 bytes, well within BORROW_COST_REGION_SIZE (4096).
 *
 * `out` must be a valid, in-bounds, writable capability over the region base
 * (the reclaimed LINEAR handle the borrow loop hands back covers exactly this).
 */
static inline void fpga_write_results(void *out, unsigned long iters,
                                      unsigned long empty, unsigned long raw,
                                      unsigned long borrow, unsigned long copy,
                                      unsigned long copy_bytes,
                                      unsigned long copy2,
                                      unsigned long copy2_bytes) {
  volatile unsigned long *r = (volatile unsigned long *)out;
  r[BORROW_COST_SLOT_ITERS] = iters;
  r[BORROW_COST_SLOT_EMPTY] = empty;
  r[BORROW_COST_SLOT_RAW] = raw;
  r[BORROW_COST_SLOT_BORROW] = borrow;
  r[BORROW_COST_SLOT_COPY] = copy;
  r[BORROW_COST_SLOT_COPY_BYTES] = copy_bytes;
  r[BORROW_COST_SLOT_COPY2] = copy2;
  r[BORROW_COST_SLOT_COPY2_BYTES] = copy2_bytes;
}

#endif
