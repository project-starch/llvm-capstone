/* The spatial arm of the nested-allocator pair: a buffer overflow from one live
   sub-allocated block into the next, with nothing freed. Parameterised by
   NEST_SPATIAL_OFFSET on the command line, so this ONE file is both arms:

     DOMAIN_EXTRA_CFLAGS=-DNEST_SPATIAL_OFFSET=64    RUNG_NAME=nestspat_in
     DOMAIN_EXTRA_CFLAGS=-DNEST_SPATIAL_OFFSET=1088  RUNG_NAME=nestspat_oob

   The runner builds the native oracle at the same -D, so each arm is compared
   against its own baseline rather than the header default. See nestalloc_kernel.h
   for what a green rung means here, which is the opposite of the usual. */
#include "nestalloc_kernel.h"

void domain_main(unsigned *res, unsigned func) {
    (void)func;
    /* Reached only if the store was allowed to complete. For the in-region arm
       that is the finding; for the out-of-region arm it would mean the control
       failed and the whole comparison is void. */
    *res = nest_spatial_run();
}
