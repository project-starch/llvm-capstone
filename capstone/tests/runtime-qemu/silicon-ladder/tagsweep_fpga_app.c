/* Board arm. LADDER_NO_MINSTRET because the perf harness's own instrumentation -- the extra
 * store through the shared-region capability at 0x200/0x208 -- was traced as the cause of the
 * retracted xg* bit-27 corruption. An instrument under suspicion must not be inside a probe
 * measuring the thing it perturbs. */
#define LADDER_NO_MINSTRET 1
#include "tagsweep_kernel.h"
#define LADDER_COMPUTE tagsweep_compute
#include "ladder_perf_domain.h"
