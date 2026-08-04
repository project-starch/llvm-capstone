#include "cp384_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp384_compute(); }
