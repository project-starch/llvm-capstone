/* QEMU variant of the s06aggcap rung. QEMU preserves tags through an aggregate copy, so a
   result other than 15 here means the rung itself is wrong, not the silicon. */
#include "s06aggcap_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06aggcap_compute(); }
