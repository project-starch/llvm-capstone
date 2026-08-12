/* QEMU variant of the s06agg rung: same kernel, writes only *res (4 B) because the QEMU harness
   hands the domain an 8-byte shared region. QEMU preserves all 128 bits of an untagged ldc/stc
   round trip, so it CANNOT exhibit S-06 -- which makes this the positive control for the repro's
   own logic: if QEMU does not return 3, the test is wrong, not the silicon. */
#include "s06agg_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06agg_compute(); }
