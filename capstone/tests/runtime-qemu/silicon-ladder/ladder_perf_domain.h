#ifndef LADDER_PERF_DOMAIN_H
#define LADDER_PERF_DOMAIN_H
/* Board-perf variant of a silicon-ladder rung's domain_main.
 *
 * The QEMU ladder's <rung>_app.c does `*res = <rung>_compute()`. For the FPGA
 * perf run we additionally record the workload's cycle cost, so this domain_main
 * brackets the compute with mcycle reads and writes both values into the shared
 * region the controller reads back:
 *   res[0] = retval  (the checksum; controller checks it == the native oracle)
 *   res[1] = cycles  (mcycle delta across <rung>_compute() only)
 *   res[2] = 0xD09E  (ran-marker, so the controller can tell a real run from
 *                     an all-zero region)
 *
 * The entry glue (start-gp-captable-generic.S) delivers the shared region cap as
 * domain_main's first argument, exactly as the QEMU path does, so the same
 * gp-captable silicon build works on both. `res` is that argument cap (reached
 * directly, not via gp); the rung's own globals are reached via gp[i] as usual.
 *
 * mcycle vs rdcycle: the board GATES the unprivileged `cycle` counter
 * (counteren.CY off for the domain), so we read the M-mode `mcycle` CSR (0xB00),
 * which the on-board setup leaves domain-readable (confirmed by the borrow-cost
 * board runs). See tests/rtl-smoke/fpga_instrument.h for the full rationale.
 *
 * Usage (per rung): `#define LADDER_COMPUTE <rung>_compute` then
 * `#include "ladder_perf_domain.h"`, after the rung's kernel header. */

#ifndef LADDER_COMPUTE
#error "define LADDER_COMPUTE to the rung's compute function before including"
#endif

static inline unsigned long ladder_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = ladder_rd_mcycle();
  unsigned v = LADDER_COMPUTE();
  unsigned long c1 = ladder_rd_mcycle();
  res[0] = (unsigned long)v;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
#endif
