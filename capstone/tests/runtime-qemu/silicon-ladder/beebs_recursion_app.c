/* Domain entry: run BEEBS recursion and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_recursion_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rec_compute(); }
