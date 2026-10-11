/* Domain entry: ctrsanity length-series point, N=20000. */
#include "ctrsanity100k_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
