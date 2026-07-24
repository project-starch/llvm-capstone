/* Domain entry: run BEEBS prime and return the checksum through `res`.
   Built with -capstone-gp-captable + shrink-off. */
#include "beebs_prime_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = prime_compute(); }
