#include "xgy_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgy_compute(); }
