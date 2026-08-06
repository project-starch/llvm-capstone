#include "fdreg_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = fdreg_compute(); }
