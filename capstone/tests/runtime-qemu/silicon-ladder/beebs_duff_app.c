/* Domain entry: run BEEBS duff and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_duff_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = duff_compute(); }
