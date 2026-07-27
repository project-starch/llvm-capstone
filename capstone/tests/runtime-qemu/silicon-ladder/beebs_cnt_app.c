/* Domain entry: run BEEBS cnt and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_cnt_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cnt_compute(); }
