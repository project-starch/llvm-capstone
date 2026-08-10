#include "s06agg_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06agg_compute(); }
