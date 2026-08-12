/* FPGA domain for rung 'trapctl': does the in-domain trap handler installed by
   INTERP_DOMAIN_MTVEC actually convert a capability fault into a return on this bitstream?

   MUST be built with INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1. Without it the glue installs
   no vector, the deliberate fault wedges, and the rung reports nothing -- which would look
   exactly like the handler failing. See trapctl_kernel.h for the res[0] verdict table; 0x7A05
   is the pass.

   domain_main is written out here rather than reused from ladder_perf_domain.h because this
   rung's whole method is the ORDER of the stores: each sentinel must reach the shared region
   before the step it labels, and the mcycle bracketing in the shared header would sit between
   the last sentinel and the fault. */
#include "trapctl_kernel.h"

void domain_main(unsigned long *res, unsigned func)
{
  (void)func;
  trapctl_run((volatile unsigned long *)res);
}
