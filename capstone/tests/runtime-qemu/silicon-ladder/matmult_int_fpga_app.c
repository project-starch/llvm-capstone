/* FPGA perf domain for silicon-ladder rung 'matmult_int': measure mcycle around the
   compute and write retval + cycles into the shared region (see
   ladder_perf_domain.h). Same gp-captable silicon build as the QEMU rung. */
#include "matmult_int_kernel.h"
#define LADDER_COMPUTE mm_compute
#include "ladder_perf_domain.h"
