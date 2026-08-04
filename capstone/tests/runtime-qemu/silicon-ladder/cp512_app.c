#include "cp512_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp512_compute(); }
