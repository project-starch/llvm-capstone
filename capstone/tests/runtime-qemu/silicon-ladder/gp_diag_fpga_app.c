/* FPGA diagnostic domain for the gp-captable silicon miscompute.
 *
 * Unlike the perf rungs (which fold everything into res[0] via ladder_perf_domain.h),
 * this one writes each probe's RAW value into its own slot, so one board run
 * localizes the failing mechanism instead of returning a single opaque checksum.
 *
 *   res[0] = checksum over all probes (so the usual oracle gate still works)
 *   res[1] = cycles (mcycle delta across the probe sweep)
 *   res[2] = 0xD09E ran-marker
 *   res[3+p] = raw value of probe p  (p = 0..GPD_NPROBE-1; see gp_diag_kernel.h)
 *
 * ladder_perf_ctl prints res[3..11] as dbg0..dbg8. */
#include "gp_diag_kernel.h"

static inline unsigned long gpd_rd_mcycle(void) {
  unsigned long v;
  __asm__ volatile("csrr %0, mcycle" : "=r"(v));
  return v;
}

/* STRAIGHT-LINE on purpose -- do NOT reintroduce a driving loop here.
 * v1 drove the probes from `for (p = 0; p < GPD_NPROBE; p++)`, and on silicon
 * that loop ran exactly ONE iteration (retval == FNV(H0,[0x5A5A]) exactly), so
 * every slot after the first read 0 and told us nothing about its own probe.
 * Unrolled, each slot is written by code that cannot be truncated by the very
 * bug under investigation. GPD_STORE keeps it to one line per probe. */
#define GPD_STORE(p) res[3 + (p)] = (unsigned long)gpd_probe(p)

void domain_main(unsigned long *res, unsigned func) {
  (void)func;
  unsigned long c0 = gpd_rd_mcycle();
  GPD_STORE(0);  GPD_STORE(1);  GPD_STORE(2);  GPD_STORE(3);
  GPD_STORE(4);  GPD_STORE(5);  GPD_STORE(6);  GPD_STORE(7);
  GPD_STORE(8);  GPD_STORE(9);  GPD_STORE(10);
  unsigned long c1 = gpd_rd_mcycle();
  /* Fold the slots back OUT of the shared region rather than re-running the
     probes (P8's local static is stateful). This fold is a loop -- deliberately:
     it runs AFTER every raw slot is already stored, so even if the loop truncates
     (the very bug under investigation) the dbg0..dbg10 values stay valid and only
     res[0] is wrong. Unrolling it cost ~2.4 KiB of .text and blew the monitor's
     4 KiB PCC window, which is a hard link-time limit here. */
  unsigned h = 2166136261u;
  for (int i = 0; i < GPD_NPROBE; i++) {
    unsigned v = (unsigned)res[3 + i];
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  res[0] = (unsigned long)h;
  res[1] = c1 - c0;
  res[2] = 0xD09EUL;
}
