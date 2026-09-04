/* The static-global arm. One file, both arms, selected on the command line:
     DOMAIN_EXTRA_CFLAGS=-DNEST_GLOBAL_OFFSET=0   RUNG_NAME=nestglob_in
     DOMAIN_EXTRA_CFLAGS=-DNEST_GLOBAL_OFFSET=64  RUNG_NAME=nestglob_oob
   The runner builds each arm's native oracle at the same -D. See nestalloc_kernel.h
   for why the second arm's outcome is genuinely open rather than a formality. */
#include "nestalloc_kernel.h"

void domain_main(unsigned *res, unsigned func) {
    (void)func;
    *res = nest_global_run();
}
