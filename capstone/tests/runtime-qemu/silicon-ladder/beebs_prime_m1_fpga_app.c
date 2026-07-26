/* Bisection variant 1 of the beebs_prime instrumentation (LADDER_INSTR_MODE=1):
   two stores of a constant to res[65] (the phase marker), nothing else.

   Mode 4 (all three constructs) miscomputes on silicon; mode 0 (none) returns the
   oracle. These isolate which single construct is the trigger. Same kernel, same
   oracle, same build path as beebs_prime -- only domain_main differs. */
#define LADDER_INSTR_MODE 1
#include "beebs_prime_kernel.h"
#define LADDER_COMPUTE prime_compute
#include "ladder_perf_domain.h"
