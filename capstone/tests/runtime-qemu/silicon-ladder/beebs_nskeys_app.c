#include "beebs_nskeys_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = nskeys_compute();}
