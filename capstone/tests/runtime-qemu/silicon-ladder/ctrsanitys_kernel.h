#ifndef CTRSANITYS_KERNEL_H
#define CTRSANITYS_KERNEL_H
/* 1/100th the work of ctrsanity, same kernel -- the SHORT counter-sanity control.
 *
 * WHY THIS EXISTS (2026-09-22). `ctrsanity` is the ladder's control: identical
 * code on both halves, so it must read 1.000x and anything else is the
 * instrument. On the BASELINE half it cannot do that job, and the 2026-09-22
 * sweep showed why in the runner's own evidence column: `clean = 1/15` in every
 * one of the three counter-probe runs, with the minimum instret varying by 6,415
 * between them. At ~500,000 instructions the kernel never completes a single
 * uninterrupted pass in Linux userspace, so its floor is never reached and its
 * "minimum" is still contaminated. A control whose own reading is noise cannot
 * certify anything.
 *
 * The other direction fails too. `beebs_janne` runs clean (15/15) but retires
 * only ~200 instructions, so the handful of bracket instructions that differ
 * between the two halves is ~6% of the measurement -- it reads 0.938x, below
 * 1.0, which the ABI cannot produce.
 *
 * So the control has to sit in the window between "long enough that bracket
 * scaffolding is negligible" and "short enough to run between timer interrupts".
 * The 2026-09-22 data brackets that window: 7,272 instructions read 14/15 clean,
 * 25,666 read 2/15. N=1000 puts this rung near 5,000 instructions -- inside the
 * clean side of the bracket, with scaffolding down to a fraction of a percent.
 *
 * Read it with ctrsanity and ctrsanity4, which are 100x and 400x this one: three
 * lengths separate a PROPORTIONAL counter effect from a FIXED one, and this end
 * of the ladder is the only end the baseline half can measure cleanly.
 */
#define CTRSANITY_N 1000L
#include "ctrsanity_kernel.h"
#endif
