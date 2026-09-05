/* QEMU half of the r14lp rung, generated 2026-09-05 from r14lp_fpga_app.c for the bug sweep:
   the same kernel and compute function, reporting through the 4-byte QEMU result slot. */
#include "r14lp_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = r14lp_compute(); }
