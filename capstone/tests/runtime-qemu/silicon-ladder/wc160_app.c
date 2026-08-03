#include "wc160_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func;*res=wc160_compute();}
