/* Domain entry: exercise the capability-aware setjmp/longjmp and return its result. */
#include "nlrjmp_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = nlrjmp_compute(); }
