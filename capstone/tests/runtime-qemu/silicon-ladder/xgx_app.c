#include "xgx_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgx_compute(); }
