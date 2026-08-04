#include "rv1000_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rv1000_compute(); }
