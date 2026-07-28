#include "beebs_aha_mont64_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = mont_compute();}
