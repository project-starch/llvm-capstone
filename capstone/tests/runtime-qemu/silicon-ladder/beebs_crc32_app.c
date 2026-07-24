/* Domain entry: run BEEBS crc32 and return the crc through the shared region cap
   `res`. Built with -capstone-gp-captable + shrink-off. */
#include "beebs_crc32_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = crc_compute(); }
