#include "xgk_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgk_compute(); }
