/* QEMU half of the beebs_janne rung, generated 2026-09-05 from beebs_janne_fpga_app.c for the bug sweep:
   the same kernel and compute function, reporting through the 4-byte QEMU result slot. */
#include "beebs_janne_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = jc_compute(); }
