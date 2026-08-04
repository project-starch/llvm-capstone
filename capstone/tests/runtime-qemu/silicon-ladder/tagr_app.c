#include "tagr_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = tagr_compute(); }
