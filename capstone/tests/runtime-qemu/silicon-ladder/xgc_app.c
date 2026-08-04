#include "xgc_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgc_compute(); }
