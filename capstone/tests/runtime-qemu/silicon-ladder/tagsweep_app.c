/* QEMU arm: SEED=0, because op_helper.c:719 asserts on a type query of an untagged value. */
#define TAGSWEEP_SEED 0u
#include "tagsweep_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = tagsweep_compute(); }
