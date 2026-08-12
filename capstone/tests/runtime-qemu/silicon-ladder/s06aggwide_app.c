/* QEMU variant: validates the rung's own logic. QEMU clears tags on plain stores, so it would
   catch a copy lowered to plain stores, but models neither S-06 nor the LDC linear-clear. */
#include "s06aggwide_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06aggwide_compute(); }
