#include "gpn16_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpn16_compute(); }
