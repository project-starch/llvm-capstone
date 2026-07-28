#include "reentry_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = reentry_compute();}
