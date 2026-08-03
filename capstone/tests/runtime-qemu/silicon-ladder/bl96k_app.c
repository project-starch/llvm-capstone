#include "bl96k_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func;*res=bl96k_compute();}
