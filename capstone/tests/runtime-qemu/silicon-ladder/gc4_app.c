#include "gc4_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = gc4_compute(); }
