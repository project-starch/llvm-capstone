/* QEMU variant: same kernel, writes only *res (4 B). NOTE: QEMU is not expected to reproduce the
   fault -- it has no D-cache model and no shadow-tag memory, so the eviction step is inert there.
   A QEMU pass proves the rung is WELL-FORMED, not that the hypothesis is false. */
#include "s06rev_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06rev_compute(); }
