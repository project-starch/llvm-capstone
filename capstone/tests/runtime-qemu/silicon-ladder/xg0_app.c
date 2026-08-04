#include "xg0_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xg0_compute(); }
