#include "gpbg_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = gpbg_compute(); }
