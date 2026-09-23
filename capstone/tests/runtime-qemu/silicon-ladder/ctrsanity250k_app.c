/* Domain entry: ctrsanity length-series point, N=50000. */
#include "ctrsanity250k_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
