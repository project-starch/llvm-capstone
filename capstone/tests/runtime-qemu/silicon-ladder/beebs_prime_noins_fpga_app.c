/* CONTROL variant of the beebs_prime perf rung: identical in every way except that
   domain_main carries NO minstret instrumentation (-DLADDER_NO_MINSTRET, i.e. the
   pre-2026-07-26 body: mcycle only, no phase slot, no instret slot).
 
   Why it exists: adding that instrumentation -- four instructions, none of them
   inside the computation -- flipped beebs_prime from returning the correct oracle
   to miscomputing on silicon. Run side by side with the instrumented rung in one
   session, this turns "it changed between two runs" into a controlled A/B, which
   is the only way to attribute the flip to the instrumentation rather than to
   anything else that moved between those runs. Same kernel, same oracle. */
#define LADDER_NO_MINSTRET 1
#include "beebs_prime_kernel.h"
#define LADDER_COMPUTE prime_compute
#include "ladder_perf_domain.h"
