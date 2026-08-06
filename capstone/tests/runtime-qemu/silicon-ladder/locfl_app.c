#include "locfl_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = locfl_compute(); }
