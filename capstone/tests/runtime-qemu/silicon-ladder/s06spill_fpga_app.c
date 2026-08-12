/* FPGA domain for rung 's06spill': does a capability spilled to the stack survive reload?
   65535 = every slot survived (spilling is sound); anything else names the failing slots.
   See s06spill_kernel.h for why this separates a spill defect from image perturbation. */
#include "s06spill_kernel.h"
#define LADDER_COMPUTE s06spill_compute
#include "ladder_perf_domain.h"
