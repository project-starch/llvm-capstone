/* FPGA domain for rung 's07chase': do back-to-back DEPENDENT capability loads and count how many
   come back NOT_CAP. 0 = this shape does not reproduce S-07; n>0 = the minimal reproducer.
   See s07chase_kernel.h. */
#include "s07chase_kernel.h"
#define LADDER_COMPUTE s07chase_compute
#include "ladder_perf_domain.h"
