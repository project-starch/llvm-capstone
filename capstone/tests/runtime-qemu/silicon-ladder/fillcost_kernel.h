#ifndef FILLCOST_KERNEL_H
#define FILLCOST_KERNEL_H
/* Silicon-ladder rung: what does the RECLAIM FILL actually cost?
 *
 * WHY THIS RUNG EXISTS. The monitor's reclaim (R-30/R-31 firmware half) overwrites a revoked
 * region before reusing it: one capability store per 16 bytes, 256 of them for the 4 KiB every
 * region in this tree is. Whether that is affordable was the open question in the reclaim
 * decision, and the honest answer was a bracket -- roughly 3 % to 25 % of the boundary path --
 * whose whole width came from ONE unmeasured quantity: what 4 KiB of capability stores costs on
 * this silicon. Everything else in that estimate is measured.
 *
 * WHY IT DOES NOT NEED THE FLASH, which is the point of measuring it this way. The reclaim PATH
 * cannot be exercised on the current bitstream: revoke there returns LINEAR for a writable region
 * (the R-31 defect), so the monitor's UNINIT guard never fires. But the fill's PHYSICAL WORK --
 * 256 capability stores walking a page -- is a pure memory-system quantity and does not depend on
 * the capability's type at all. So it is measurable today, on the flashed bitstream, as an
 * ordinary rung.
 *
 * READ BOTH COUNTERS. ladder_perf_domain.h brackets with mcycle AND minstret, which is exactly the
 * split this question needs: the fill's instruction count is known a priori (~4 per store), so
 * cycles-over-instructions is the number that was missing.
 *
 * PREDICTED READING, written before the run: 256 stores, so ~1024 instructions. Optimistic bound
 * ~1000 cycles if the stores retire near back to back; ceiling ~14,400 from the measured 1024-byte
 * copy rate of 3.52 cyc/byte extrapolated to 4096 -- which OVERSTATES it, because a copy loads and
 * stores where a fill only stores. A reading near the ceiling makes the reclaim a quarter of the
 * boundary path; near the floor it is a few percent.
 *
 * A READING NEAR ZERO IS NOT GOOD NEWS -- it means the loop was optimised away. The buffer is
 * volatile and the return value depends on it, so it cannot be, but check the instruction count
 * against ~1024 before believing any cycle figure.
 *
 * WHY void ** AND NOT memset: subscripting a capability-typed double pointer steps 16 bytes and
 * emits `stc`, which is the instruction the monitor's fill uses. A byte memset would emit scalar
 * stores and measure the wrong thing. */

#define FILLCOST_BYTES 4096
#define FILLCOST_SLOTS (FILLCOST_BYTES / 16)

static void *fillcost_buf[FILLCOST_SLOTS];

static unsigned fillcost_compute(void)
{
  volatile void **p = (volatile void **)fillcost_buf;
  unsigned i;
  unsigned r = 0;
  /* the fill itself: one capability store per 16 bytes, exactly the monitor's shape */
  for (i = 0; i < FILLCOST_SLOTS; i += 1)
    p[i] = 0;
  /* make the result depend on the fill so it cannot be discarded */
  for (i = 0; i < FILLCOST_SLOTS; i += 64)
    r += (p[i] == 0);
  return r; /* FILLCOST_SLOTS/64 = 4 */
}
#endif
