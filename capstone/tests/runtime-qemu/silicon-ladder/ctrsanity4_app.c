/* Domain entry: counter-sanity probe. */
#include "ctrsanity4_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = cs_compute(); }
