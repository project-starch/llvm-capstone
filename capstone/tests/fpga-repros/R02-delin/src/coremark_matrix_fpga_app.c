/* FPGA perf domain for silicon-ladder rung 'coremark_matrix': measure mcycle around the
   compute and write retval + cycles into the shared region (see
   ladder_perf_domain.h). Same gp-captable silicon build as the QEMU rung. */
#include "coremark_matrix_kernel.h"
#define LADDER_COMPUTE coremark_matrix_compute
#include "ladder_perf_domain.h"
