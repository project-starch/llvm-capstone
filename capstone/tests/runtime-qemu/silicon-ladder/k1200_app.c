#include "k1200_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = k1200_compute(); }
