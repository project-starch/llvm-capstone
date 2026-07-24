/* FPGA perf domain for silicon-ladder rung 'beebs_crc32': measure mcycle around the
   compute and write retval + cycles into the shared region (see
   ladder_perf_domain.h). Same gp-captable silicon build as the QEMU rung. */
#include "beebs_crc32_kernel.h"
#define LADDER_COMPUTE crc_compute
#include "ladder_perf_domain.h"
