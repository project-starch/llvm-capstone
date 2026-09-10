/* Domain entry for the QEMU half of the reclaim-fill cost rung: run the fill and return its
   check value through `res`. The board half is fillcost_fpga_app.c, which reports cycles and
   instructions as well; this one exists so the COMPUTE can be validated under emulation before
   the .dom is staged. See fillcost_kernel.h. */
#include "fillcost_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = fillcost_compute(); }
