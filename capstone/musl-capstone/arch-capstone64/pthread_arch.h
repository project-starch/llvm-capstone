/* THE THREAD POINTER IS A CAPABILITY, and upstream's accessor returns an integer.
 *
 * arch/capstone64 is a copy of arch/riscv64 with this directory overlaid, so
 * every file we do not override is inherited verbatim. This one was, and it
 * carried a defect that only shows up on a capability target:
 *
 *     static inline uintptr_t __get_tp(void)          // upstream riscv64
 *     {
 *         uintptr_t tp;
 *         __asm__ __volatile__("mv %0, tp" : "=r"(tp));
 *         return tp;
 *     }
 *
 * `mv` is `addi rd, rs, 0`, an ORDINARY integer move, and uintptr_t is 64 bits
 * here while a capability is 128. So the tag was dropped on the way out and
 * every use of __pthread_self() computed an address rather than a pointer.
 * musl's own macro then does pointer arithmetic on it:
 *
 *     #define __pthread_self() ((pthread_t)(__get_tp() - sizeof(struct __pthread) ...))
 *
 * which became `addi` on an untagged value, and the first dereference took a
 * cause-24. Found by mruby's gem test suite: a failing clock_gettime reached an
 * error path, strerror read CURRENT_LOCALE out of the TLS block, and the domain
 * halted with
 *
 *     Cap mem access requires capability: rs1 = x13, imm = 0
 *     mv   a3, tp        <- the tag goes here
 *     addi a3, a3, -0x60
 *     ldc  a1, 0x0(a3)   <- and this is where it is noticed
 *
 * SAME REASONING AS syscall_arg_t, which had to become `void *` for exactly this
 * reason (see syscall_arch.h): there is no capability-carrying INTEGER type on
 * this target, so anything that must keep a tag has to be a pointer type. That
 * conclusion was drawn for syscall arguments and never applied here.
 *
 * `movc` and `void *`: the capability move keeps the tag, and returning a
 * pointer means musl's own TP_ADJ arithmetic selects to cincoffsetimm instead of
 * addi. Verified in the generated assembly, not assumed.
 *
 * NOT a thread-safety change. A domain is single-threaded; this is about the
 * PROVENANCE of one register, not about concurrency.
 */
static inline void *__get_tp(void)
{
	void *tp;
	__asm__ __volatile__("movc %0, tp" : "=r"(tp));
	return tp;
}

#define TLS_ABOVE_TP
#define GAP_ABOVE_TP 0

#define DTP_OFFSET 0x800

#define MC_PC __gregs[0]
