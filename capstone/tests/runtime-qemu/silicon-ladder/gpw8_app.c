#include "gpw8_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpw8_compute(); }
