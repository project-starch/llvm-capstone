#include "beebs_nsflat_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = nsflat_compute();}
