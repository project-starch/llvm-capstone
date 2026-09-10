/* Rung app for the reclaim-fill cost measurement. See fillcost_kernel.h for what it measures and
 * why it does not need the new bitstream. */
#include "fillcost_kernel.h"
#define LADDER_COMPUTE fillcost_compute
#include "ladder_perf_domain.h"
