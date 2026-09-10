/* Domain entry for the QEMU half of the fillwarm rung. See fillwarm_kernel.h. */
#include "fillwarm_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fillwarm_compute(); }
