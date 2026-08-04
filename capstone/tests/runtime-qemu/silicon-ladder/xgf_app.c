#include "xgf_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgf_compute(); }
