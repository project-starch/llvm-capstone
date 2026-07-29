/* FPGA perf domain for the descriptor-stress rung: exercises every glue path SQLite
   needs (zero-fill, bulk copy, byte tail, >2040 B global, private .L symbol) in the
   smallest domain that can. See gpstress_kernel.h for why each global is there. */
#include "gpstress_kernel.h"
#define LADDER_COMPUTE gpstress_compute
#include "ladder_perf_domain.h"
