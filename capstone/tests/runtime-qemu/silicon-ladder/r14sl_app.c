#include "r14sl_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = r14sl_compute(); }
