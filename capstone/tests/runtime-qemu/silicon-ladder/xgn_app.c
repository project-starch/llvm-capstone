#include "xgn_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgn_compute(); }
