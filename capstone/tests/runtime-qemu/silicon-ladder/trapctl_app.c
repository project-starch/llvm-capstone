/* Domain entry for the QEMU half of the trap-handler control. Identical in body to
   trapctl_fpga_app.c -- unlike most rungs there is no board-only instrumentation to strip, because
   this rung's whole method is the ORDER of the sentinel stores and nothing may sit between the last
   sentinel and the deliberate fault.

   It exists so the rung can be verified under emulation before a boot is spent on it, which is what
   the staged-run preflight requires. MUST be built with INTERP_DOMAIN_MTVEC=1: without the in-domain
   trap vector the deliberate fault wedges instead of returning, and a wedge here is indistinguishable
   from the handler being broken -- which is the very distinction this rung exists to make.
   See trapctl_kernel.h for the res[0] verdict table; 0x7A05 is the pass. */
#include "trapctl_kernel.h"

void domain_main(unsigned long *res, unsigned func)
{
  (void)func;
  trapctl_run((volatile unsigned long *)res);
}
