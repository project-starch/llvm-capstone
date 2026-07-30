#include "gpw4_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpw4_compute(); }
