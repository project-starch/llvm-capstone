#include "xgh_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgh_compute(); }
