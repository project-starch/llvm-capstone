#include "rv600_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = rv600_compute(); }
