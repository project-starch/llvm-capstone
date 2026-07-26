/* FPGA/board entry for the v4 per-element readback dump.
 *
 *   res[0]   = straight-line sum of the second-pass window read (oracle gate)
 *   res[1]   = cycles, res[2] = 0xD09E ran-marker
 *   res[3+k] = raw readback k, k = 0..32   (see gp_diag4_kernel.h)
 *   res[40..48) = the seeded data window
 *
 * Needs a controller that shares a real 4 KiB region (rtl-smoke/ladder_perf_ctl.c)
 * and LADDER_DBG_SLOTS >= 33. Validate with rtl-smoke/run-ladder-perf-qemu.sh,
 * which runs that same controller under QEMU. */
#include "gp_diag4_kernel.h"

static inline unsigned long gpd4_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = gpd4_rd_mcycle();
  unsigned long sum = gpd4_run(res);
  unsigned long c1 = gpd4_rd_mcycle();
  res[0] = sum;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
