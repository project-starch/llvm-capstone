#include "s06copy_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06copy_compute(); }
