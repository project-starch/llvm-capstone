#ifndef CTRSANITY250K_KERNEL_H
#define CTRSANITY250K_KERNEL_H
/* The ctrsanity kernel at N=50000 (~250,000 instructions): one point of the phase-6 length
 * series (2026-09-23). The control reads 1.045x at 5k instructions and 1.167x at both
 * 500k and 2M, and nothing was sampled in between. This rung, with ctrsanity20k /
 * ctrsanity100k / ctrsanity250k, places the transition. Identical code on both halves;
 * see ctrsanity_kernel.h for the kernel and ladder-revival-2026-09-22.prereg.md (phase 6)
 * for the predictions this rung was built to test. */
#define CTRSANITY_N 50000L
#include "ctrsanity_kernel.h"
#endif
