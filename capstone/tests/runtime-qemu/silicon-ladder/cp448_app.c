#include "cp448_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp448_compute(); }
