#include "locnc_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = locnc_compute(); }
