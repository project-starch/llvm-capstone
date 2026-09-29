/* CINCOFFSET and SCC on an untagged value: what the ISA does today, and the probe the decision in
 * docs/plans/2026-09-24-scc-cincoffset-untagged.md predicts against. Built once per case (-DCASE),
 * because a trap ends the domain.
 *
 *   case 0  both instructions on a real capability: the control, and the proof that the two
 *           hand-encoded instructions are the ones meant (the stores land where they should)
 *   case 1  cincoffset on the integer 0x5000, by 8
 *   case 2  scc on the integer 0x5000, to 0x6000
 *
 * Today, spec and RTL: cases 1 and 2 raise Unexpected operand type (24). Under CHERI's rule they
 * would print 0x5008 and 0x6000. A QEMU without the scc fix aborts the whole machine at case 2.
 */
#include <stdio.h>

static char buf[64];

int main(void)
{
#if CASE == 0
	void *p = buf, *q, *r;
	unsigned long off = 8, to = __builtin_capstone_cap_get_cursor(buf) + 16;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x0c, %0, %1, %2" : "=r"(q) : "r"(p), "r"(off));
	__asm__ volatile(".insn r 0x5b, 0x1, 0x05, %0, %1, %2" : "=r"(r) : "r"(p), "r"(to));
	*(char *)q = 'q';
	*(char *)r = 'r';
	printf("ARITH-CASE 0 cinc=%d scc=%d\n", buf[8] == 'q', buf[16] == 'r');
#elif CASE == 1
	unsigned long v = 0x5000, off = 8, res = 0;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x0c, %0, %1, %2" : "=r"(res) : "r"(v), "r"(off));
	printf("ARITH-CASE 1 result=0x%lx\n", res);
#elif CASE == 2
	unsigned long v = 0x5000, to = 0x6000, res = 0;
	__asm__ volatile(".insn r 0x5b, 0x1, 0x05, %0, %1, %2" : "=r"(res) : "r"(v), "r"(to));
	printf("ARITH-CASE 2 result=0x%lx\n", res);
#endif
	fflush(stdout);
	return 0;
}
