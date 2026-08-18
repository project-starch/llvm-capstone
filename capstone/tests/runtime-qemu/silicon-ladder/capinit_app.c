#include "capinit_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = capinit_compute(); }
