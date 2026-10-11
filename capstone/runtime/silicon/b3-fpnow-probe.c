/* B3j (docs/plans/b0-silicon-delegated-runtime.md): the R-52 residual on silicon, checked against a reference that
 * does not use the FPU. A NATIVE program (the board's glibc), run by b0run.sh's b3-fpnow rung.
 *
 * On caplifive_r51r52_0568f93a9 the oracle harness's now() = tv_sec + tv_nsec / 1e9 returned 0.09..1.99 while the
 * integer clock read ~240 s (the R-52 retraction, dev 23213fb4a701). This probe evaluates that expression in four
 * EXACT instruction orders, in asm, and checks every result by decoding the double's bits with INTEGER arithmetic:
 *   A  now()'s order: fld 1e9; fcvt.d.l nsec; fdiv.d; fcvt.d.l sec issued while the fdiv is in flight; fadd.d
 *   B  the B3c arith loop's order: both fcvt.d.l before the fdiv.d, then fadd.d
 *   C  A with 40 nops after the fdiv.d (the fdiv completes before the second fcvt.d.l issues)
 *   D  A with the seconds fcvt.d.l moved before the fdiv.d (the operand ready, the fadd still behind the fdiv)
 * Inputs: live CLOCK_MONOTONIC readings, and a fixed pair (239 s, 702477440 ns) so timing cannot vary the operands.
 * A result passes if its integer part equals tv_sec and its fraction is within 2 ns of tv_nsec.
 * Exit status: a bitmask, bit n set if shape n (A=0 .. D=3) had any mismatch. The printed lines are the reading. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static const double ONE_E9 = 1e9;
/* Each returns the result's raw bits. a0 = nsec, a1 = sec, a2 = &ONE_E9. */
uint64_t shapeA(long nsec, long sec, const double *k);
uint64_t shapeB(long nsec, long sec, const double *k);
uint64_t shapeC(long nsec, long sec, const double *k);
uint64_t shapeD(long nsec, long sec, const double *k);
__asm__(
".text\n"
".globl shapeA\n.type shapeA,@function\nshapeA:\n"
"  fld fa4, 0(a2)\n  fcvt.d.l fa0, a0\n  fdiv.d fa0, fa0, fa4\n  fcvt.d.l fa5, a1\n  fadd.d fa0, fa0, fa5\n"
"  fmv.x.d a0, fa0\n  ret\n"
".globl shapeB\n.type shapeB,@function\nshapeB:\n"
"  fld fa3, 0(a2)\n  fcvt.d.l fa5, a0\n  fcvt.d.l fa4, a1\n  fdiv.d fa5, fa5, fa3\n  fadd.d fa5, fa5, fa4\n"
"  fmv.x.d a0, fa5\n  ret\n"
".globl shapeC\n.type shapeC,@function\nshapeC:\n"
"  fld fa4, 0(a2)\n  fcvt.d.l fa0, a0\n  fdiv.d fa0, fa0, fa4\n  .rept 40\n  nop\n  .endr\n  fcvt.d.l fa5, a1\n"
"  fadd.d fa0, fa0, fa5\n  fmv.x.d a0, fa0\n  ret\n"
".globl shapeD\n.type shapeD,@function\nshapeD:\n"
"  fld fa4, 0(a2)\n  fcvt.d.l fa0, a0\n  fcvt.d.l fa5, a1\n  fdiv.d fa0, fa0, fa4\n  fadd.d fa0, fa0, fa5\n"
"  fmv.x.d a0, fa0\n  ret\n");

/* Integer-only decode of a positive normal double into whole seconds and nanoseconds of fraction. */
static int decode(uint64_t b, long *sec, long *ns)
{
	int e = (int)((b >> 52) & 0x7ff);
	if (b >> 63 || e == 0 || e == 0x7ff) return 0;
	uint64_t m = (b & ((1ull << 52) - 1)) | (1ull << 52);
	int sh = 1075 - e;                       /* value = m * 2^-sh */
	if (sh <= 0 || sh > 120) return 0;
	unsigned __int128 mm = m;
	unsigned __int128 ip = sh >= 64 ? 0 : (mm >> sh);
	unsigned __int128 frac = mm - (ip << sh);
	*sec = (long)ip;
	*ns = (long)((frac * 1000000000ull + (((unsigned __int128)1) << (sh - 1))) >> sh);
	return 1;
}

static int check(uint64_t bits, long sec, long nsec)
{
	long s, n;
	if (!decode(bits, &s, &n)) return 0;
	if (n == 1000000000L) { s++; n = 0; }
	long d = n - nsec;
	return s == sec && d <= 2 && d >= -2;
}

typedef uint64_t (*shape_fn)(long, long, const double *);

int main(void)
{
	setvbuf(stdout, NULL, _IONBF, 0);
	/* The checker's own control: the correct double for (239 s, 702477440 ns), computed on the host, must pass; the
	 * value the board's now() returned on 0568f93a9 (0.2390064) must fail. */
	int ctl_ok = check(0x406df67ab1f7df4aull, 239L, 702477440L), ctl_bad = check(0x3fce97c2ffc470deull, 239L, 702477440L);
	printf("B3j checker control: correct accepted %d, wrong rejected %d\n", ctl_ok, !ctl_bad);
	if (!ctl_ok || ctl_bad) { printf("B3j probe rc=255 (checker void)\n"); return 255; }
	const shape_fn fn[4] = {shapeA, shapeB, shapeC, shapeD};
	const char *nm = "ABCD";
	int rc = 0;
	for (int k = 0; k < 4; k++) {
		long bad_live = 0, bad_fixed = 0, n_live = 20000, n_fixed = 20000;
		long fs = 0, fn_ = 0; uint64_t fb = 0; int first = 1;
		for (long i = 0; i < n_live; i++) {
			struct timespec t;
			clock_gettime(CLOCK_MONOTONIC, &t);
			uint64_t b = fn[k](t.tv_nsec, t.tv_sec, &ONE_E9);
			if (!check(b, t.tv_sec, t.tv_nsec)) {
				if (first) { fs = t.tv_sec; fn_ = t.tv_nsec; fb = b; first = 0; }
				bad_live++;
			}
		}
		for (long i = 0; i < n_fixed; i++) {
			uint64_t b = fn[k](702477440L, 239L, &ONE_E9);
			if (!check(b, 239L, 702477440L)) {
				if (first) { fs = 239; fn_ = 702477440L; fb = b; first = 0; }
				bad_fixed++;
			}
		}
		printf("B3j %c: live %ld of %ld wrong, fixed %ld of %ld wrong", nm[k], bad_live, n_live, bad_fixed, n_fixed);
		if (!first) {
			long s = -1, n = -1;
			decode(fb, &s, &n);
			printf("; first: sec %ld nsec %ld -> %016llx (decodes to %ld s + %ld ns)", fs, fn_, (unsigned long long)fb, s, n);
		}
		printf("\n");
		if (bad_live || bad_fixed) rc |= 1 << k;
	}
	printf("B3j probe rc=%d\n", rc);
	return rc;
}
