/* FPGA domain for rung 's06lcc': silicon confirmation that LCC's TYPE query is TOTAL on the
   resident bitstream, and that the query-based S-06 repair works on real hardware.
   Expect retval 171. See s06lcc_kernel.h for what each digit means. */
#include "s06lcc_kernel.h"
#define LADDER_COMPUTE s06lcc_compute
#include "ladder_perf_domain.h"
