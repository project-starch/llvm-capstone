#include "k800_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = k800_compute(); }
