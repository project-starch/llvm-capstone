#include "gpn4_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpn4_compute(); }
