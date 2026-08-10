#include "s06fix_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06fix_compute(); }
