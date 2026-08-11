/* QEMU variant of the s06lcc rung: same kernel, but writes only *res (4 B), because the QEMU
   harness hands the domain an 8-byte shared region and the _fpga build's 24-byte write faults
   before it can report anything. Validates the COMPUTE; the board image is built from the same
   source at the same parameterisation. */
#include "s06lcc_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s06lcc_compute(); }
