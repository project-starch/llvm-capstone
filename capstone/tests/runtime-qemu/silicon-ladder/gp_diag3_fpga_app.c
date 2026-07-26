/* FPGA/board entry for the v3 shared-region diagnostic rung.
 *
 *   res[0]   = FNV fold over the raw slots (the harness's oracle gate)
 *   res[1]   = cycles (mcycle delta)
 *   res[2]   = 0xD09E ran-marker
 *   res[3+p] = raw value of probe p, p = 0..8   (see gp_diag3_kernel.h)
 *   res[32..40), res[40..48) = the seeded data windows the probes read/write
 *
 * The 4 KiB region gives 512 slots, so the windows are far inside bounds.
 * NOTE this rung only works with a controller that shares a real 4 KiB region
 * (rtl-smoke/ladder_perf_ctl.c). The plain capstone-test `call_dom` path passes
 * a pointer to a SINGLE unsigned on the monitor's stack, so it must not be used
 * here -- writing res[3..] there would smash the monitor. That is why there is
 * no gp_diag3_app.c; QEMU validation runs ladder_perf_ctl itself under QEMU
 * (rtl-smoke/run-ladder-perf-qemu.sh), which is exact parity with the board. */
#include "gp_diag3_kernel.h"

static inline unsigned long gpd3_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = gpd3_rd_mcycle();
  unsigned h = gpd3_run(res);
  unsigned long c1 = gpd3_rd_mcycle();
  res[0] = (unsigned long)h;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
