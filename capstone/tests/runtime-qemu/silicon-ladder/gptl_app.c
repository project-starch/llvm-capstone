#include "gptl_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gptl_compute(); }
