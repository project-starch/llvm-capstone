#include "gpw16_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpw16_compute(); }
