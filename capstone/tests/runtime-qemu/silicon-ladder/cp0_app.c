#include "cp0_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp0_compute(); }
