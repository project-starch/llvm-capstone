/* One silicon-ladder rung's kernel, compiled as PLAIN RISC-V (no capabilities).
 *
 * This is the BASELINE half of the spatial-safety overhead measurement: the very
 * same <rung>_kernel.h that the capability domain runs (via <rung>_fpga_app.c +
 * ladder_perf_domain.h), compiled by the SAME clang at the SAME -O level, but for
 * -target riscv64 instead of -target capstone64. Everything else about the
 * measurement -- board, clock, DRAM, the counter read -- is held fixed, so the
 * capability-vs-baseline cycle ratio isolates the capability ABI + its hardware
 * enforcement rather than a compiler difference.
 *
 * Compiled once per rung into its own translation unit (never all seven into one)
 * because the kernel headers are independent BEEBS/CoreMark/RV8 sources that reuse
 * common file-scope names; separate TUs keep them from colliding and keep each
 * rung's codegen identical to the single-kernel domain build.
 *
 *   cc -DLADDER_KERNEL_HDR='"beebs_prime_kernel.h"' \
 *      -DLADDER_COMPUTE=prime_compute -DLADDER_EXPORT=base_beebs_prime ...
 */
#ifndef LADDER_KERNEL_HDR
#error "define LADDER_KERNEL_HDR to the rung's kernel header"
#endif
#ifndef LADDER_COMPUTE
#error "define LADDER_COMPUTE to the rung's compute function"
#endif
#ifndef LADDER_EXPORT
#error "define LADDER_EXPORT to the exported wrapper name"
#endif

#include LADDER_KERNEL_HDR

unsigned LADDER_EXPORT(void) { return LADDER_COMPUTE(); }
