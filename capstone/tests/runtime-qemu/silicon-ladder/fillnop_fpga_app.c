/* Board half of the fill-cost control rung: brackets the same loop with mcycle and minstret so
 * the pair difference is the capability store's marginal cost. See fillcost_kernel.h. */
#include "fillnop_kernel.h"
#define LADDER_COMPUTE fillnop_compute
#include "ladder_perf_domain.h"
