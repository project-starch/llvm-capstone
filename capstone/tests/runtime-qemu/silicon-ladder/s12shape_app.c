/* QEMU-side variant of rung 's12shape'. Writes *res only (4 bytes), per the ladder convention;
   the board variant is s12shape_fpga_app.c and writes res[0..2].

   QEMU does not reproduce S-12, so the expected QEMU outcome is 0x5122 -- the loop completing
   without a fault. That is what the .qemu-pass marker records: that the rung RUNS, not that the
   hardware behaviour was reproduced. A QEMU pass here says nothing about silicon and is not
   evidence about the mechanism. */
#define S12SHAPE_RES0_ONLY 1
#include "s12shape_kernel.h"

void domain_main(unsigned long *res, unsigned func)
{
  (void)func;
  s12shape_run((volatile unsigned long *)res);
}
