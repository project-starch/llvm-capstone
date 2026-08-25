/* Board arm for the S-12 minimal repro. LADDER_NO_MINSTRET: the perf harness's own store
 * through the shared-region capability must not sit inside a probe that is measuring stores. */
#define LADDER_NO_MINSTRET 1
#include "s12_kernel.h"
#define LADDER_COMPUTE s12_compute
#include "ladder_perf_domain.h"
