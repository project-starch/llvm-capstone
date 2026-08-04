#include "rv1400_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rv1400_compute(); }
