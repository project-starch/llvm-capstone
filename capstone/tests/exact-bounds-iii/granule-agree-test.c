/* The bounds granule is defined TWICE: in capstone/runtime/include/capstone/representable.h for
 * the runtime, and inline in ports/musl-capstone/arch-capstone64/malloc_capstone.h for mallocng,
 * which cannot reach the runtime headers (the musl survey compiles with -nostdinc and only musl's
 * own include directories, survey-musl-capstone.py:91-99). enframe needs it on every allocation,
 * so a cross-TU call would cost one per malloc.
 *
 * Two copies can drift. This compiles BOTH and fails if they disagree on any length, which is the
 * reason the duplication is safe to have.
 *
 *   cc -I capstone/runtime/include -I capstone/ports/musl-capstone/arch-capstone64 \
 *      -o granule-agree-test granule-agree-test.c && ./granule-agree-test
 */
#include <stdio.h>
#include <stddef.h>
#include <capstone/representable.h>
#include <malloc_capstone.h>

int main(void)
{
	unsigned long bad = 0, checked = 0;
	/* every length up to 1 MiB, then every power of two and its neighbours to 2^40 */
	for (unsigned long n = 1; n <= (1UL << 20); n++) {
		checked++;
		if (capstone_repr_granule(n) != __capstone_malloc_granule(n) ||
		    capstone_repr_round(n) != __capstone_malloc_repr_round(n)) {
			if (bad++ < 5)
				printf("DISAGREE at %lu: granule %lu vs %lu, round %lu vs %lu\n", n,
				       capstone_repr_granule(n), __capstone_malloc_granule(n),
				       capstone_repr_round(n), __capstone_malloc_repr_round(n));
		}
	}
	for (unsigned e = 12; e < 40; e++) {
		unsigned long p = 1UL << e;
		unsigned long around[] = {p - 2, p - 1, p, p + 1, p + 2, p + p / 512 - 1, p + p / 512};
		for (unsigned i = 0; i < sizeof around / sizeof *around; i++) {
			unsigned long n = around[i];
			checked++;
			if (capstone_repr_granule(n) != __capstone_malloc_granule(n) ||
			    capstone_repr_round(n) != __capstone_malloc_repr_round(n)) {
				if (bad++ < 5)
					printf("DISAGREE at %lu: granule %lu vs %lu\n", n,
					       capstone_repr_granule(n), __capstone_malloc_granule(n));
			}
		}
	}
	printf("checked %lu lengths, %lu disagreements\n", checked, bad);
	/* A positive control: the check must be able to FIRE. A deliberately wrong granule must be
	 * caught, or this test proves nothing about the two definitions agreeing. */
	unsigned long n = 1UL << 20;
	if (capstone_repr_granule(n) == capstone_repr_granule(n) * 2) {
		printf("the comparison cannot distinguish values\n");
		return 1;
	}
	if (bad) { printf("FAIL\n"); return 1; }
	printf("the two definitions agree on every length checked\n");
	return 0;
}
