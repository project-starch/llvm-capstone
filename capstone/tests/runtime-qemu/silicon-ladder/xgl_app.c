#include "xgl_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = xgl_compute(); }
