/* jmp_buf for a pure-capability domain.
 *
 * Upstream riscv64 is `unsigned long __jmp_buf[26]`: 8-byte slots, 208 bytes,
 * sized for ra, sp, s0-s11 and the twelve callee-saved FP registers. Here ra,
 * sp and s0-s11 are capabilities, so each needs a 16-byte slot stored with stc,
 * and the buffer must itself be 16-aligned or the store loses the tag silently.
 * Fourteen capability slots are 224 bytes; 28 unsigned longs is exactly that.
 * No FP slots: the domain is soft-float and has no FP registers to preserve.
 *
 * Overlaid onto the copy of arch/riscv64 like syscall_arch.h and
 * pthread_arch.h, so `diff -r arch/riscv64 arch/capstone64` still shows the
 * whole delta. setjmp.h itself is upstream's and unchanged. */
typedef unsigned long __jmp_buf[28] __attribute__((aligned(16)));
