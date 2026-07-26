/* QEMU-side entry for the gp-captable diagnostic rung: same probes as
   gp_diag_fpga_app.c, returning the fold so run-ladder-qemu.sh can gate it
   against the native oracle. The board variant additionally writes each probe's
   raw value into its own result slot. Built with -capstone-gp-captable +
   shrink-off, exactly like the perf rungs. */
#include "gp_diag_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = gpd_compute(); }
