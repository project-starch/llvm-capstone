/* FPGA perf domain for silicon-ladder rung 'rv8_primes': measure mcycle around the
   compute and write retval + cycles into the shared region (see
   ladder_perf_domain.h). Same gp-captable silicon build as the QEMU rung. */
#include "rv8_primes_kernel.h"
#define LADDER_COMPUTE primes_compute
#include "ladder_perf_domain.h"
