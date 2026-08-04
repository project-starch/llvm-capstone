#include "tagf_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = tagf_compute(); }
