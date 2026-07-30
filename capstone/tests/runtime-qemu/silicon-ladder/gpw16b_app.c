#include "gpw16b_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpw16b_compute(); }
