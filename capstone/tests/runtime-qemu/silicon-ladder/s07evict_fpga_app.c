/* FPGA domain for rung 's07evict': spill 16 capabilities, EVICT them from L1, reload and query.
   65535 = eviction is harmless too; anything else = the minimal reproducer. See the kernel header. */
#include "s07evict_kernel.h"
#define LADDER_COMPUTE s07evict_compute
#include "ladder_perf_domain.h"
