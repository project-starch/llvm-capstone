#include "k800n_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = k800n_compute(); }
