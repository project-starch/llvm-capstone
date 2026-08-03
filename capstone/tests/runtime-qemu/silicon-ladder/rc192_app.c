#include "rc192_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rc192_compute(); }
