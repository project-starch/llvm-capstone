/* sweep KLO KHI R: for n0 in [2^k - 4096, 2^k) step 16 and k in [KLO, KHI], R rounds each (mallocng
 * cycles the offset within the slot on reuse): p = malloc(n0), then realloc to n1 in {usable(p), 2^k}
 * when n1 > n0. Before each realloc, a pair whose CURRENT base is not a multiple of the granule of
 * repr_round(n1) is printed as CAND: an in-place growth there publishes an inexact child. After the
 * realloc, INPLACE-CAND marks a candidate that stayed in place (the trap would have fired before). */
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned long gran(unsigned long len)
{
	return len < 4096 ? 1 : 1UL << (63 - __builtin_clzl(len) - 9);
}
static unsigned long rround(unsigned long len)
{
	unsigned long r = (len + gran(len) - 1) / gran(len) * gran(len);
	while (r % gran(r)) r = (r + gran(r) - 1) / gran(r) * gran(r);
	return r;
}
int main(int argc, char **argv)
{
	int klo = atoi(argv[1]), khi = atoi(argv[2]), rounds = atoi(argv[3]);
	unsigned long cands = 0, inplace = 0, calls = 0;
	for (int k = klo; k <= khi; k++)
		for (unsigned long n0 = (1UL << k) - 4096; n0 < (1UL << k); n0 += 16)
			for (int r = 0; r < rounds; r++)
				for (int w = 0; w < 2; w++) {
					char *p = malloc(n0);
					if (!p) { printf("SWEEP malloc NULL %lu\n", n0); return 2; }
					unsigned long u = malloc_usable_size(p), a = __builtin_capstone_cap_get_cursor(p);
					unsigned long n1 = w ? (1UL << k) : u;
					if (n1 <= n0) { free(p); continue; }
					int cand = a % gran(rround(n1)) != 0;
					if (cand) { cands++; printf("CAND k=%d n0=%lu n1=%lu usable=%lu a%%g=%lu g=%lu\n", k, n0, n1, u, a % gran(rround(n1)), gran(rround(n1))); fflush(stdout); }
					char *q = realloc(p, n1); calls++;
					if (!q) { printf("SWEEP realloc NULL %lu %lu\n", n0, n1); return 3; }
					if (__builtin_capstone_cap_get_cursor(q) == a) { inplace++; if (cand) printf("INPLACE-CAND n0=%lu n1=%lu\n", n0, n1); }
					free(q);
				}
	printf("SWEEP DONE k=%d..%d calls=%lu inplace=%lu cands=%lu\n", klo, khi, calls, inplace, cands);
	return 0;
}
