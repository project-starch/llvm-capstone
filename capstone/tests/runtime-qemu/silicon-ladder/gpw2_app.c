#include "gpw2_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpw2_compute(); }
