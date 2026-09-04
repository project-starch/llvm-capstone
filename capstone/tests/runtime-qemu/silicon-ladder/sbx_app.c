/* QEMU half of the R-20 repro rung (fpga-repros/R20-stc-rs1-cursor-forward-x10/src): the same
   kernel as sbx_fpga_app.c, reporting through the 4-byte QEMU result slot. */
#include "sbx_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = sbx_compute(); }
