/* QEMU arm of the s07evict rung. See s07evict_kernel.h. */
#include "s07evict_kernel.h"
static unsigned compute(void) { return s07evict_compute(); }
void domain_main(unsigned *res, unsigned func) { (void)func; *res = compute(); }
