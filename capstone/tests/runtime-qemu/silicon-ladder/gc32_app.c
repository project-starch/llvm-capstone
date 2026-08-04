#include "gc32_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = gc32_compute(); }
