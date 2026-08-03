#include "rc128_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rc128_compute(); }
