#include "gpn8_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpn8_compute(); }
