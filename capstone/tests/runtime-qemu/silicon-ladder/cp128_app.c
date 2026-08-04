#include "cp128_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp128_compute(); }
