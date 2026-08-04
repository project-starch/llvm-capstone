#include "cp16_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp16_compute(); }
