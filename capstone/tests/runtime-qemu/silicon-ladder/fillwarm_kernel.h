#ifndef FILLWARM_KERNEL_H
#define FILLWARM_KERNEL_H
/* Does a WARM region cost less to fill? The audit of §7e refused the assumption in both directions:
 * this cache is write-through with NO write-allocate, so a store to a resident line still writes
 * through and "the buffer was cold, so 23.6 cyc/store is a ceiling" does not follow.
 *
 * This rung runs the SAME loop TWICE over the same buffer, so
 *     fillwarm cycles - fillcost cycles  =  the cost of a pass over a region just filled.
 * Near fillcost's own figure means cold is not the driver and the caveat should be DELETED, not
 * softened. Materially below it means warm is cheaper and the ceiling reading was right after all.
 * Informative either way, which is why it is worth an arm. */
#define FILL_PASSES 2
#define FILL_TAG 2048
#include "fillcost_kernel.h"
#define fillwarm_compute fillcost_compute
#endif
