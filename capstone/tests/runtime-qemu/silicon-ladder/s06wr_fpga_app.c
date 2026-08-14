/* FPGA domain for rung 's06wr': does a capability spilled to the stack survive reload?
   65535 = every slot survived (spilling is sound); anything else names the failing slots.
   See s06wr_kernel.h for why this separates a spill defect from image perturbation. */
#include "s06wr_kernel.h"
#define LADDER_COMPUTE s06wr_compute
#include "ladder_perf_domain.h"
