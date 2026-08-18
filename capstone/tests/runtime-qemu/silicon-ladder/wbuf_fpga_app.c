/* Board arm. LADDER_NO_MINSTRET for the same reason as tagsweep: the perf harness's own
 * store through the shared-region capability must not sit inside a probe measuring stores. */
#define LADDER_NO_MINSTRET 1
#include "wbuf_kernel.h"
#define LADDER_COMPUTE wbuf_compute
#include "ladder_perf_domain.h"
