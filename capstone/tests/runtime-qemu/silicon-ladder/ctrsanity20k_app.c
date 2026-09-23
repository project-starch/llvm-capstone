/* Domain entry: ctrsanity length-series point, N=4000. */
#include "ctrsanity20k_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
