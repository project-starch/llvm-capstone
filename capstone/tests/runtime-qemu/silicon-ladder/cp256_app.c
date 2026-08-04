#include "cp256_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp256_compute(); }
