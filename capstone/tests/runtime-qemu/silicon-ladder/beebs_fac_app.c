/* Domain entry: run BEEBS fac and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_fac_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fac_compute(); }
