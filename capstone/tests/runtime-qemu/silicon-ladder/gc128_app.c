#include "gc128_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = gc128_compute(); }
