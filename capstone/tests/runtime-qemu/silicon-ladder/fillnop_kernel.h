#ifndef FILLNOP_KERNEL_H
#define FILLNOP_KERNEL_H
/* The matched control for `fillcost`: the SAME loop with the capability store replaced by a nop.
 * Same iteration count, same pointer walk, same branch, same instruction count -- one instruction
 * different. See fillcost_kernel.h for why the pair is the measurement and neither arm alone is.
 * Expect 256 (loop ran, nothing stored); fillcost expects 768. */
#define FILL_PAYLOAD "addi x0, x0, 0\n"
#define FILL_STORES 0
#include "fillcost_kernel.h"
#define fillnop_compute fillcost_compute
#endif
