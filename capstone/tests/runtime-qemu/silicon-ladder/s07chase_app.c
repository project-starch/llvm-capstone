/* QEMU arm of the s07chase rung. See s07chase_kernel.h for why this shape. */
#include "s07chase_kernel.h"
static unsigned compute(void) { return s07chase_compute(); }
void domain_main(unsigned *res, unsigned func) { (void)func; *res = compute(); }
