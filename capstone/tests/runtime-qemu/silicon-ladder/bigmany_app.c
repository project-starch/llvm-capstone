#include "bigmany_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = bigmany_compute(); }
