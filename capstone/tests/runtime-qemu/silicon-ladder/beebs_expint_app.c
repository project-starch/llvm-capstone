#include "beebs_expint_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = expint_compute();}
