#include "locnp_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = locnp_compute(); }
