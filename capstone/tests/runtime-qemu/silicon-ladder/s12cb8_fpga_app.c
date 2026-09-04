/* FPGA domain for rung 's12shape': does the S-12 instruction shape fault inside a REAL capability
   domain on silicon, where bare-metal Verilator says it does not?

   MUST be built with INTERP_DOMAIN_MTVEC=1, so a fault returns a code instead of wedging the core.
   Without it "no return" and "no fault" are indistinguishable, which is what made every earlier
   reconstruction of this shape unreadable.

   See s12shape_kernel.h for the res[0] verdict table. A returned word beginning 0xF is the glue's
   trap report: mcause in bits 27..22, (mepc - _start) >> 2 in bits 21..0. */
#define S12SHAPE_CAPBURST 8
#include "s12shape_kernel.h"

void domain_main(unsigned long *res, unsigned func)
{
  (void)func;
  s12shape_run((volatile unsigned long *)res);
}
