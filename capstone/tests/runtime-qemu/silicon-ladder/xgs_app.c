#include "xgs_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgs_compute(); }
