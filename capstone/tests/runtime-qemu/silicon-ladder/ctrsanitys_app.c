/* Domain entry: short counter-sanity control. */
#include "ctrsanitys_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
