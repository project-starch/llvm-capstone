/* QEMU variant of 's06sfix': same kernel, writes only *res (4 B) because the QEMU harness's
   shared region is 8 B and the _fpga build's 24 B write faults on res[1]. */
#include "s06scale_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06scale_fix(); }
