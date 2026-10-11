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

/* LADDER_PAD=K (opt-in, 2026-09-23, layout-randomised ladder, phase 10 of ladder-revival-2026-09-22):
   a FILE-SCOPE block at the top of this translation unit's .text -- `.p2align 6` then K four-byte
   nops (.4byte, so RVC cannot shrink them). Every function in the object (kernel, its non-inlined
   helpers, the wrapper) shifts by 4*K bytes from a 64-byte-aligned start, independently of any other
   rung's object, and NOTHING is executed. Phases 8-9 showed cycle ratios move with layout on this
   silicon, so each kernel is measured at several K and reported as a median and band. Undefined =>
   nothing emitted, byte-identical. (A first design padded inside the function behind a `j`; in the
   baseline image every earlier rung's function grew too, so a rung's loop moved 16 bytes per K and
   never changed residue -- caught at the desk, replaced by this.) */
#ifdef LADDER_PAD
#  define LADDER_PAD_STR2(x) #x
#  define LADDER_PAD_STR(x) LADDER_PAD_STR2(x)
__asm__(".pushsection .text\n .p2align 6\n .rept " LADDER_PAD_STR(LADDER_PAD) "\n .4byte 0x00000013\n .endr\n .popsection\n");
#endif
#include LADDER_KERNEL_HDR

unsigned LADDER_EXPORT(void) { return LADDER_COMPUTE(); }
