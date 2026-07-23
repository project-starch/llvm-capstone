/* Domain entry: compute the matmul checksum and return it through the shared
   region cap `res`. Built with -capstone-gp-captable + shrink-off. */
#include "matmult_int_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = mm_compute(); }
