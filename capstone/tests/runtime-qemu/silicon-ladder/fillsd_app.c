/* Domain entry for the QEMU half of the fillsd rung. See fillsd_kernel.h. */
#include "fillsd_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fillsd_compute(); }
