/* __intcap end to end (intcap plan, Phases B and C), on QEMU.
 *
 * D is __uintcap_t (VARIANT "intcap") or unsigned long (VARIANT "uptr", the control).
 * Every value goes through volatile function pointers and across calls, so it is
 * copied, spilled and reloaded; under CAPSTONE_MOVC_NULL_SCALAR=1 that also
 * exercises the live-source copy rule for integers held in capability registers.
 *
 * The integer cases (int, arith, cmp, switch) must print "ok" in both variants.
 * The pointer cases (ptr, ptrarith) dereference a pointer that went through D:
 * with __uintcap_t it keeps its capability and reads the byte; with unsigned long
 * it is untagged and the first dereference faults, ending the domain. That fault
 * is the control's expected result, so the pointer cases run last.
 */
#include <stdio.h>

#ifdef USE_INTCAP
typedef __uintcap_t D;
#else
typedef unsigned long D;
#endif

static D id(D v) { return v; }
static D (*volatile idp)(D) = id;
static int bad;

static void check(const char *what, unsigned long got, unsigned long want)
{
	if (got == want)
		printf("INTCAP " VARIANT " %s ok\n", what);
	else {
		printf("INTCAP " VARIANT " %s BAD got=0x%lx want=0x%lx\n", what, got, want);
		bad++;
	}
	/* The control ends in a fault, which loses whatever stdout still buffers. */
	fflush(stdout);
}

struct box { long pad; D d; };
static struct box (*volatile boxp)(struct box);
static struct box box_id(struct box b) { return b; }

static char buf[64];

int main(void)
{
	boxp = box_id;
	for (int i = 0; i < 64; i++)
		buf[i] = (char)(0x40 + i);

	/* int: an integer held in D survives copies, calls and spills. */
	D x = 0x5000;
	D y = idp(x), z = idp(x);
	check("int", (unsigned long)idp(y) + (unsigned long)idp(z) + (unsigned long)x, 0xf000);

	/* arith: +, *, & run on the value (for __intcap: on the address, then the
	   address-replacing dispatch takes the integer path). */
	D a = idp(x);
	a = a + 8;
	a = a * 2;
	a = a & (D)0xfffffff0UL;
	check("arith", (unsigned long)a, 0xa010);

	/* cmp: comparison and difference of two D holding pointers. */
	D p0 = (D)(void *)buf, p1 = idp(p0) + 16;
	check("cmp", (p0 < p1) + (unsigned long)(p1 - p0), 17);

	/* switch on a D. */
	int s;
	switch (idp(x)) {
	case 0x4000: s = 1; break;
	case 0x5000: s = 2; break;
	default: s = 3; break;
	}
	check("switch", s, 2);
	printf("INTCAP " VARIANT " integers done bad=%d\n", bad);
	fflush(stdout);

	/* ptr: a pointer stored in a struct as D, passed by value through a call,
	   converted back and dereferenced. */
	struct box b = { 7, (D)(void *)&buf[5] };
	b = boxp(b);
	check("ptr", *(char *)(void *)idp(b.d), 0x45);

	/* ptrarith: a pointer in D moved by arithmetic, then dereferenced (for
	   __intcap: the dispatch keeps the authority and moves the cursor). */
	D q = idp((D)(void *)buf);
	q = q + 20;
	check("ptrarith", *(char *)(void *)q, 0x54);

	printf("INTCAP " VARIANT " END bad=%d\n", bad);
	return bad;
}
