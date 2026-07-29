#include "gpn32_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpn32_compute(); }
