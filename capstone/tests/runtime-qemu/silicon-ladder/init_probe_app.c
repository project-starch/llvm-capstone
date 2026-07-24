#include "init_probe_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = ip_compute(); }
