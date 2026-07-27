/* Domain entry: run BEEBS fibcall and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_fibcall_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fibcall_compute(); }
