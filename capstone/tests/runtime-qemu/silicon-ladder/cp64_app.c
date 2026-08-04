#include "cp64_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cp64_compute(); }
