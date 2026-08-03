#include "bl32k_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func;*res=bl32k_compute();}
