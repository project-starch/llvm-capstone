/* Domain entry for the QEMU half of the fill-cost CONTROL rung. See fillnop_kernel.h. */
#include "fillnop_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fillnop_compute(); }
