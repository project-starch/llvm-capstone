/* Bisection variant 6: one store of a constant to res[32] (offset 0x100, MID).

   Mode 3 (one store to res[64], offset 0x200) FAILS on silicon; mode 2 (104 B of
   CSR reads, no store) PASSES. So the trigger is an extra store through the
   shared-region capability. This variant asks whether the OFFSET is what matters --
   res[0..2] at 0x0/0x8/0x10 are stored by the passing control too. */
#define LADDER_INSTR_MODE 6
#include "beebs_prime_kernel.h"
#define LADDER_COMPUTE prime_compute
#include "ladder_perf_domain.h"
