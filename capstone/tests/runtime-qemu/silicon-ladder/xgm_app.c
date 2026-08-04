#include "xgm_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgm_compute(); }
