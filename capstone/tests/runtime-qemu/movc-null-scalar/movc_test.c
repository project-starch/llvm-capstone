/* MOVC of a non-capability source under capstone-qemu's CAPSTONE_MOVC_NULL_SCALAR.
 *
 * MOVC on the RTL writes cnull over its source unless the source is a non-linear
 * capability, so an INTEGER source is zeroed; capstone-qemu keeps it by default
 * (Q-04). run.sh boots this domain twice, with the switch off and on.
 *
 *   probe   two hand-written movc of one integer: the first copies it, and on the
 *           RTL also zeroes it, so the second copies zero. b=5 c=5 means the source
 *           survived (QEMU's default), b=5 c=0 that it was nulled (the RTL). This is
 *           the SQLite port's stage-50 probe, and it depends on no compiler.
 *   c32     what the compiler makes of the SQLite port's setupLookaside (C-32),
 *           reduced: an integer turned into a pointer on one arm, passed to a call,
 *           then tested and read back after the join. At -O2 the call argument is
 *           a `movc a0, sN` whose source sN is read again afterwards, so where the
 *           source is nulled the join reads null and the function returns 1 instead
 *           of the address -- the lookaside silently off. Which of the two it prints
 *           under the switch says whether the compiler still emits that copy.
 *   iconv   musl's iconv_open, reduced (C-32's second instance): a descriptor that is
 *           an integer made into a pointer inside a callee, kept by the caller and
 *           passed to three calls. Where each copy into a0 zeroes the source, the
 *           second and third calls see null: n=1 instead of 3.
 *
 * run.sh builds this file twice: VARIANT "rule" with the compiler's default (the
 * live-source copy rule, CapstoneLiveSourceCopy) and VARIANT "keep" with
 * +movc-keeps-integer-source, which emits a plain movc for every copy. With the
 * switch on, "keep" must lose the value (the positive control) and "rule" must not.
 */
#include <stdio.h>

typedef unsigned long uptr;

static int called;
static int record(void *p)
{
	called += p != 0;
	return 0;
}

__attribute__((noinline)) uptr lookaside_shape(uptr v, void *buf, int (*fp)(void *))
{
	void *p;
	if (buf == 0) {
		p = (void *)v;
		if (p)
			fp(p);
	} else {
		p = buf;
	}
	if (p)
		return (uptr)p;
	return 1;
}

__attribute__((noinline)) void *open_desc(uptr v)
{
	return (void *)(v << 16 | 1);
}
__attribute__((noinline)) int use_desc(void *d)
{
	return d != 0;
}
int main(void)
{
	unsigned long src = 5, b = 0, c = 0;
	/* movc rd, rs1: opcode 0x5b, funct3 1, funct7 0x0a, rs2 x0 -- the encoding the
	   stage-50 probe verified against a compiler-emitted movc. */
	__asm__ volatile(".insn r 0x5b, 0x1, 0x0a, %0, %2, x0\n\t"
	                 ".insn r 0x5b, 0x1, 0x0a, %1, %2, x0"
	                 : "=&r"(b), "=&r"(c), "+r"(src));
	printf("MOVC-PROBE b=%lu c=%lu\n", b, c);

	/* volatile, so the address is an opaque integer to the optimizer. */
	static volatile uptr addr = 0x5000;
	uptr got = lookaside_shape(addr, 0, record);
	printf("MOVC-C32 " VARIANT " got=0x%lx want=0x%lx called=%d\n", got, (uptr)addr, called);
	/* Through volatile function pointers, so the optimizer can see neither callee:
	   with direct calls it folded use_desc's `d != 0` and no copy of cd was left. */
	static void *(*volatile open_fp)(uptr) = open_desc;
	static int (*volatile use_fp)(void *) = use_desc;
	void *cd = open_fp(addr);
	int n = use_fp(cd);
	n += use_fp(cd);
	n += use_fp(cd);
	printf("MOVC-ICONV " VARIANT " n=%d\n", n);
	fflush(stdout);
	return 0;
}
