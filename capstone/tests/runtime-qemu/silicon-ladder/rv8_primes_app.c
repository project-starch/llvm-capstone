/* Domain entry: run RV8 primes and return the largest prime through the shared
   region cap `res`. Built with -capstone-gp-captable + shrink-off. */
#include "rv8_primes_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = primes_compute(); }
