/* QEMU variant. QEMU clears a granule's tag on a plain store, so it would catch a spill lowered
   to plain stores; it returns 65535 when spilling is capability-grained, as it should be. */
#include "s06spill_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06spill_compute(); }
