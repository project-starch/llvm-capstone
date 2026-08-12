/* FPGA domain for rung 's06aggwide': guarded aggregate copy with capabilities in NON-ZERO
   granules, 4 granules, into a STACK destination. 255 = fully correct. See the kernel header
   for the bit map and for why each axis is untested by s06agg / s06aggcap. */
#include "s06aggwide_kernel.h"
#define LADDER_COMPUTE s06aggwide_compute
#include "ladder_perf_domain.h"
