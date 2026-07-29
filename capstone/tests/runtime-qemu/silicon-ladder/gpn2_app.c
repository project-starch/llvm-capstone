#include "gpn2_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpn2_compute(); }
