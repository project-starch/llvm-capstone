/* The globals carve on silicon (docs/plans/b0-silicon-delegated-runtime.md, "the pool carve"): read back the bounds of
 * this program's own globals, as the core decoded them, against each global's size. A delegated application built by
 * build-b0-hello.sh (B0_APP=carve-witness), twice: B0_GLUE_CARVE=align (the split carve) and B0_GLUE_CARVE=pool.
 *
 * Under the full bounds scheme (the FP bitstream (b) and the Sublet bitstream) the split carve rounds the stack
 * remainder's top up into the global just carved, so a global's capability reaches into its neighbour by up to the
 * remainder's granule (check-repr.py OVERLAP); the pool carve gives each global exactly its own (padded) storage.
 * On a cursorless bitstream both are exact. check-repr.py's replay predicts every line below for a given base.
 *
 * Each line: the global's index, its size, then base and end of the capability the program holds for it, as offsets
 * from the global's own address. Nothing is dereferenced past a global's size. Exit status: the number of globals whose
 * capability reaches below their address or more than 15 bytes past their size (16-byte storage rounding is not
 * overlap); 0 = every global separated. */
#include <stdio.h>

#define BASE(p) __builtin_capstone_cap_get_base(p)
#define END(p) __builtin_capstone_cap_get_end(p)
#define CUR(p) __builtin_capstone_cap_get_cursor(p)

/* Sizes that span the granule classes: under 4 KiB (exact at 16 B), 4-8 KiB (granule 8), 8-16 KiB (16), 64 KiB (128). */
static char g0[24] = {1};
static char g1[100] = {1};
static char g2[5000] = {1};
static char g3[40] = {1};
static char g4[9000] = {1};
static char g5[7] = {1};
static char g6[65536] = {1};
static char g7[200] = {1};

static void *opaque(void *p)
{
	__asm__ volatile("" : "+r"(p));
	return p;
}

static int line(int i, void *p, unsigned long size)
{
	p = opaque(p);
	long b = (long)(BASE(p) - CUR(p)), e = (long)(END(p) - CUR(p));
	int over = b < 0 || e > (long)((size + 15) & ~15ul);
	printf("CW %d: size %lu base %+ld end %+ld %s\n", i, size, b, e, over ? "BEYOND" : "own");
	return over;
}

int main(void)
{
	int n = 0;
	n += line(0, g0, sizeof g0);
	n += line(1, g1, sizeof g1);
	n += line(2, g2, sizeof g2);
	n += line(3, g3, sizeof g3);
	n += line(4, g4, sizeof g4);
	n += line(5, g5, sizeof g5);
	n += line(6, g6, sizeof g6);
	n += line(7, g7, sizeof g7);
	printf("CW rc=%d\n", n);
	return n;
}
