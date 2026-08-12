/* FPGA domain for rung 's06rev': minimal repro for the second S-06 blocker -- does a revocation
   node survive eviction from L1? See s06rev_kernel.h for the mechanism and how to read the
   outcome. Expected to WEDGE if the hypothesis holds, so run it LAST in its boot. */
#include "s06rev_kernel.h"
#define LADDER_COMPUTE s06rev_compute
#include "ladder_perf_domain.h"
