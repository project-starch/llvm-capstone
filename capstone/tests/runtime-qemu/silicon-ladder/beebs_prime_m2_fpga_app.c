/* Bisection variant 2 of the beebs_prime instrumentation (LADDER_INSTR_MODE=2):
   two `csrr minstret` reads, with NO region store at all.

   Mode 4 (all three constructs) miscomputes on silicon; mode 0 (none) returns the
   oracle. These isolate which single construct is the trigger. Same kernel, same
   oracle, same build path as beebs_prime -- only domain_main differs. */
#define LADDER_INSTR_MODE 2
#include "beebs_prime_kernel.h"
#define LADDER_COMPUTE prime_compute
#include "ladder_perf_domain.h"
