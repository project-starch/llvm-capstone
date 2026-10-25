/* R-54 on silicon (ISSUES R-54, docs/plans/b0-silicon-delegated-runtime.md): the full bounds scheme does not
 * renormalise, so a capability whose span, rounded to its granule, reaches the next power of two decodes about half
 * short -- on 0568f93a9 once its cursor leaves the base (the cursorless branch covers cursor == base only).
 *
 * A delegated application, built by build-b0-hello.sh (B0_APP=r54-bounds), run by b0run.sh. Nothing is
 * dereferenced through a capability under test: every reading is the decoded base and end, read with lcc, so every
 * arm RETURNS. Each arm narrows a capability to [s, s + n) with its cursor on the base (shrink: cursor == base, the
 * cursorless write-back), reads the bounds, moves the cursor by 8 (cincoffset: the full-branch write-back), and
 * reads them again. s is an offset from a 4 KiB-aligned anchor inside a 1 MiB heap block, so each arm's alignment
 * modulo its granule is fixed and the expected offsets are base-independent.
 *   A  n 65535 at anchor + 0     wraps: moved end +32768 on 0568f93a9
 *   B  n 65520 at anchor + 16    wraps: moved [-16, +32752)
 *   C  n  8191 at anchor + 0     wraps: moved end +4096
 *   D  n 65536 at anchor + 0     control, exactly a power of two: covers
 *   G  n 65471 at anchor + 0     control, rounds up to +65472 without crossing: covers
 *   H  malloc(65535), the allocator's own pointer, walked by 8: its expectation is R-54's predicate on its own base
 * Exit status: bit k set if arm k (A=0 B=1 C=2 D=3 G=4 H=5) read SHORT once moved; 64 if any of A..G's AT-BASE
 * readings was not exact. H's expectation depends on where the allocator put it (and QEMU never reads short), so H
 * prints the predicate beside its reading. The printed lines are the reading. */
#include <stdio.h>
#include <stdlib.h>

#define CUR(p) __builtin_capstone_cap_get_cursor(p)
#define BASE(p) __builtin_capstone_cap_get_base(p)
#define END(p) __builtin_capstone_cap_get_end(p)

/* An opaque pass: the compiler may not fold the bounds of p + 8 into those of p. */
static void *opaque(void *p)
{
	__asm__ volatile("" : "+r"(p));
	return p;
}

/* R-54's predicate: [s, s + n) with n >= 4 KiB decodes short once re-encoded in the full branch iff its span,
 * rounded out to g = 2^(E+3) with E = hb(n) - 12, reaches 2^(E+13). */
static int r54_predicts_short(unsigned long s, unsigned long n)
{
	if (n < 4096)
		return 0;
	int hb = 63 - __builtin_clzl(n);
	unsigned long g = 1ul << (hb - 12 + 3);
	unsigned long lo = s & ~(g - 1), hi = (s + n + g - 1) & ~(g - 1);
	return hi - lo >= (g << 10);
}

static int arm(const char *name, char *p, unsigned long n, int *atbase_bad)
{
	unsigned long s = CUR(p);
	char *r = opaque(__builtin_capstone_cap_shrink(p, s, s + n));
	long b0 = (long)(BASE(r) - s), e0 = (long)(END(r) - s);
	char *m = opaque(r + 8);
	long bm = (long)(BASE(m) - s), em = (long)(END(m) - s);
	int shortm = bm > 0 || em < (long)n;
	if (b0 != 0 || e0 != (long)n)
		*atbase_bad = 1;
	printf("R54 %s: s%%4096 %lu n %lu: at base [%+ld, %+ld) moved [%+ld, %+ld) %s (predicate: %s)\n", name,
	       s & 4095, n, b0, e0, bm, em, shortm ? "SHORT" : "covers", r54_predicts_short(s, n) ? "short" : "covers");
	return shortm;
}

int main(void)
{
	char *blk = malloc(1 << 20);
	if (!blk) {
		printf("R54: malloc(1 MiB) failed\n");
		return 255;
	}
	unsigned long b = CUR(blk);
	char *anchor = blk + (((b + 4095) & ~4095ul) - b);
	int bad0 = 0, rc = 0;
	rc |= arm("A", anchor + 0, 65535, &bad0) << 0;
	rc |= arm("B", anchor + 16, 65520, &bad0) << 1;
	rc |= arm("C", anchor + 0, 8191, &bad0) << 2;
	rc |= arm("D", anchor + 0, 65536, &bad0) << 3;
	rc |= arm("G", anchor + 0, 65471, &bad0) << 4;
	/* H: the allocator's own narrowed pointer (level0 narrows to exactly n), walked by 8. */
	char *h = malloc(65535);
	if (!h) {
		printf("R54: malloc(65535) failed\n");
		return 255;
	}
	unsigned long hs = CUR(h);
	long hb0 = (long)(BASE(h) - hs), he0 = (long)(END(h) - hs);
	char *hm = opaque(h + 8);
	long hbm = (long)(BASE(hm) - hs), hem = (long)(END(hm) - hs);
	int hshort = hbm > 0 || hem < 65535, hpred = r54_predicts_short(hs, 65535);
	printf("R54 H: malloc(65535) s%%4096 %lu: at base [%+ld, %+ld) moved [%+ld, %+ld) %s (predicate: %s)\n",
	       hs & 4095, hb0, he0, hbm, hem, hshort ? "SHORT" : "covers", hpred ? "short" : "covers");
	rc |= hshort << 5;
	if (bad0)
		rc |= 64;
	printf("R54 rc=%d\n", rc);
	return rc;
}
