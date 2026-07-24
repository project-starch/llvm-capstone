/* Domain entry: run BEEBS insertsort and return the checksum through the shared
   region cap `res`. Built with -capstone-gp-captable + shrink-off. */
#include "beebs_insertsort_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = is_compute(); }
