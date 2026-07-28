/* FPGA perf domain for silicon-ladder rung 'beebs_prime': measure mcycle around the
   compute and write retval + cycles into the shared region (see
   ladder_perf_domain.h). Same gp-captable silicon build as the QEMU rung. */
#include "beebs_prime_kernel.h"
#define LADDER_COMPUTE prime_compute
#include "ladder_perf_domain.h"
