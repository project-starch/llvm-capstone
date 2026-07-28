#include "rv8_sha512s_kernel.h"
void domain_main(unsigned *res, unsigned func){(void)func; *res = sha512s_compute();}
