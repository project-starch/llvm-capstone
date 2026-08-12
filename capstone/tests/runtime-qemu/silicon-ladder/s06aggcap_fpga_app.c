/* FPGA domain for rung 's06aggcap': does the guarded aggregate copy preserve a REAL CAPABILITY?
   Expect 15 with a correct guard. 12 means plain data survived but the capability did not --
   the regression s06agg structurally cannot see. See s06aggcap_kernel.h. */
#include "s06aggcap_kernel.h"
#define LADDER_COMPUTE s06aggcap_compute
#include "ladder_perf_domain.h"
