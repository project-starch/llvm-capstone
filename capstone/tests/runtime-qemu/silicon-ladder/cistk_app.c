/* The freestanding memcpy comes from the SAME file the mruby domain links, because the
   struct assignment in the kernel is lowered to a memcpy call and the copy's behaviour is
   part of what is being measured. Including it keeps this rung one translation unit, which
   the gp-captable ABI requires anyway. */
#include "../../../benchmarks/beebs/adapted/beebs_freestanding_string.c"
#include "cistk_kernel.h"
void domain_main(unsigned *res, unsigned func){ (void)func; *res = cistk_compute(); }
