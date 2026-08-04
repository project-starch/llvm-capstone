#include "xgq_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgq_compute(); }
