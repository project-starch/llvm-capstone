/* Domain entry: counter-sanity probe. */
#include "ctrsanity_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
