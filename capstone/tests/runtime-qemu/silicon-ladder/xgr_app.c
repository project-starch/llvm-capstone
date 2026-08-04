#include "xgr_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgr_compute(); }
