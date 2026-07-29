#include "gppv_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gppv_compute(); }
