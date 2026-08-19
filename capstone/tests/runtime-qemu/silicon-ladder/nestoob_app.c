/* Arm B, the POSITIVE CONTROL. Byte-identical to arm A except for the one
   constant below: the stale write is aimed past the end of the region instead of
   inside it. The hardware must trap this. If it does not, the capability checking
   is not live in this build and arm A's quiet success proves nothing at all. */
#define NEST_STALE_OFFSET (NEST_HEAP_BYTES + NEST_BLOCK)
#include "nestalloc_kernel.h"

void domain_main(unsigned *res, unsigned func) {
    (void)func;
    /* Written only if the out-of-bounds store was allowed to complete, which
       would mean the control FAILED. A trap means we never reach this line. */
    *res = nest_run();
}
