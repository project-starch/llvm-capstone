/* FPGA domain for the blob-peek probe: returns the 64-bit word the domain actually reads
   from the monitor-copied blob at offset 8 (see blobpeek_kernel.h and INTERP_DIAG_STAGE=11). */
#include "blobpeek_kernel.h"
#define LADDER_COMPUTE bp_compute
#include "ladder_perf_domain.h"
