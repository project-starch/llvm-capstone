#include "xgz_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgz_compute(); }
