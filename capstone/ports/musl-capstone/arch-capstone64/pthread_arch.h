/* Capstone pure-capability thread pointer for musl.
 *
 * Overlaid onto the copy of arch/riscv64 by prepare-musl-capstone.sh, so
 * `diff -r arch/riscv64 arch/capstone64` remains the whole delta of this port.
 *
 * WHY THIS FILE HAS TO EXIST. Upstream reads the thread pointer as an integer:
 *
 *     static inline uintptr_t __get_tp()
 *     { uintptr_t tp; __asm__ __volatile__("mv %0, tp" : "=r"(tp)); return tp; }
 *
 * On this target __UINTPTR_TYPE__ is `long unsigned int` at 64 bits and `mv` is
 * an integer move, so both the type and the instruction drop the tag. tp comes
 * back as a bare address and the first dereference through it faults with
 * cause = 24, requires capability. That is the same defect syscall_arch.h
 * carried until 2026-09-16, in the same shape, one register over.
 *
 * WHAT CHANGES, AND WHY NOTHING ELSE HAS TO. Two things: the instruction is
 * `movc`, which moves a capability, and the return type is a pointer, which is
 * the only type on this target that carries a tag (there is no __uintcap_t).
 *
 * Returning `char *` rather than `void *` is deliberate and is what lets
 * upstream's pthread_impl.h stay untouched. It computes
 *
 *     #define __pthread_self() ((pthread_t)(__get_tp() - sizeof(struct __pthread) - TP_OFFSET))
 *
 * which is integer arithmetic when __get_tp() returns an integer and POINTER
 * arithmetic when it returns char *. The pointer form is what we want: it
 * lowers to cincoffset, which moves the cursor and preserves both the bounds
 * and the tag. Measured: `movc a0, tp` followed by `cincoffsetimm a0, a0, -0x8`.
 *
 * TLS_ABOVE_TP, GAP_ABOVE_TP, DTP_OFFSET and MC_PC are upstream riscv64's and
 * are repeated here because this file replaces the whole header rather than
 * patching it. They must be kept in step with arch/riscv64/pthread_arch.h.
 */
static inline char *__get_tp(void)
{
	char *tp;
	__asm__ __volatile__("movc %0, tp" : "=r"(tp));
	return tp;
}

#define TLS_ABOVE_TP
#define GAP_ABOVE_TP 0

#define DTP_OFFSET 0x800

#define MC_PC __gregs[0]
