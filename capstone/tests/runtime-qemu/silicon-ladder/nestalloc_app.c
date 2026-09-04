/* Arm A of the nested-allocator pair: the stale write lands INSIDE the region.
   Expected result: it succeeds, and the domain returns the same value a native
   build returns, i.e. Capstone provided no temporal protection. See
   nestalloc_kernel.h for why a green rung here is the finding. */
#define NEST_STALE_OFFSET 0u
#include "nestalloc_kernel.h"

void domain_main(unsigned *res, unsigned func) {
    (void)func;
    *res = nest_run();
}
