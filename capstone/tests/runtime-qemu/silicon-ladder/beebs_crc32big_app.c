/* Domain entry: large-RO crc32 (upstream const table) via the copy path. */
#include "beebs_crc32big_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = crc_compute(); }
