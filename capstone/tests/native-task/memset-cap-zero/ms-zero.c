/* CAPSTONE_MEMSET_CAP_ZERO directed test: the runtime's memset over memory holding tagged capabilities, for zero and
 * non-zero fills, at every head offset 0..15 and lengths that cover no granule, part of one, whole granules, the
 * 64-byte loop and its tails. Each case:
 *   - poisons both words of every granule of a 20-granule buffer (0xA5.. pattern), then stores a tagged capability
 *     into every granule, so each granule's two words are a capability's;
 *   - snapshots the words, calls memset through a volatile pointer (the runtime's, never an inline expansion);
 *   - after a fence, and again after an eviction sweep (a 1 MiB walk), reads both 64-bit words of every granule with
 *     plain loads and the tag with __builtin_capstone_cap_get_tag, and requires: bytes inside [off, off + n) equal
 *     the fill, bytes of every granule the range does not touch equal the snapshot; a granule the range touches has
 *     its tag cleared; any other keeps its tag and its capability (same cursor, and it still names its cell).
 * Controls: the same readout with no memset must find every tag set (the tag read can report a set tag), and a
 * mutant (-DMS_MUTANT) whose fill stops one byte short must fail the byte check. */
#include <stdio.h>
#include <string.h>

#define TAG(p) __builtin_capstone_cap_get_tag((void *)(p))
#define BARRIER() __asm__ volatile("" ::: "memory")
#define FENCE() __asm__ volatile("fence rw, rw" ::: "memory")
#define NG 20

static int cells[NG];
static void *buf[NG] __attribute__((aligned(16)));
static unsigned long snap[2 * NG];
static unsigned char sweep[1 << 20];
static int checks, failures, cases;

static void *(*volatile memset_call)(void *, int, unsigned long) = memset;

static void check(int ok, const char *name)
{
	++checks;
	failures += !ok;
	printf("MS_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
}

static void prepare(void)
{
	volatile unsigned long *w = (volatile unsigned long *)buf;
	for (int i = 0; i < 2 * NG; ++i)
		w[i] = 0xa5a5a5a5a5a5a5a5ul ^ (unsigned long)i;
	BARRIER();
	for (int g = 0; g < NG; ++g)
		buf[g] = &cells[g];
	BARRIER();
	for (int i = 0; i < 2 * NG; ++i)
		snap[i] = ((volatile unsigned long *)buf)[i];
}

static void evict(void)
{
	volatile unsigned char *s = sweep;
	for (unsigned i = 0; i < sizeof sweep; i += 64)
		s[i] = (unsigned char)(s[i] + 1);
	FENCE();
}

/* Returns 1 if the buffer is as a fill of c over [16 + off, 16 + off + n) requires. */
static int diag_left = 6;
static int verify(unsigned off, unsigned n, int c, int filled)
{
	int ok = 1;
	const unsigned char *b = (const unsigned char *)buf, *sb = (const unsigned char *)snap;
	unsigned lo = 16 + off, hi = 16 + off + (filled ? n : 0);
	for (unsigned i = 0; i < sizeof buf; ++i) {
		/* A byte outside the range, in a granule the range touches, is not memset's: once the fill clears that
		   granule's tag, its other bytes read as the twin's raw capability encoding, not the cursor a plain load of
		   the tagged capability returned (measured with the existing memset too, 2026-11-14). */
		unsigned g0 = i / 16;
		if (!(i >= lo && i < hi) && hi > lo && 16 * g0 < hi && 16 * g0 + 16 > lo)
			continue;
		int want = i >= lo && i < hi ? (unsigned char)c : sb[i];
		if (b[i] != want && diag_left > 0) {
			--diag_left;
			printf("MS_DIAG:off=%u:n=%u:byte=%u:granule=%u:in_range=%d:got=%#x:want=%#x:tag=%d\n", off, n, i, i / 16,
			       i >= lo && i < hi, b[i], want, (int)TAG(buf[i / 16]));
		}
		ok &= b[i] == want;
	}
	for (unsigned g = 0; g < NG; ++g) {
		unsigned a = 16 * g, e = a + 16;
		int touched = hi > lo && a < hi && e > lo;
		if (touched)
			ok &= TAG(buf[g]) == 0;
		else
			ok &= TAG(buf[g]) == 1 && buf[g] == &cells[g] && *(int *)buf[g] == (int)g * 3 + 1;
	}
	return ok;
}

int main(void)
{
	static const unsigned lens[] = {0, 1, 7, 8, 15, 16, 17, 24, 31, 32, 33, 47, 48, 63, 64, 65, 79, 80, 96, 127, 128,
					129, 191, 192, 200, 255};
	static const int fills[] = {0, 0x5a};
	for (int i = 0; i < NG; ++i)
		cells[i] = i * 3 + 1;

	prepare();
	FENCE();
	check(verify(0, 0, 0, 0), "control_no_memset_tags_set");

	int fence_ok = 1, evict_ok = 1;
	for (unsigned f = 0; f < 2; ++f)
		for (unsigned off = 0; off < 16; ++off)
			for (unsigned k = 0; k < sizeof lens / sizeof lens[0]; ++k) {
				unsigned n = lens[k];
				if (16 + off + n > sizeof buf - 16)
					continue;
				prepare();
#ifdef MS_MUTANT
				memset_call((char *)buf + 16 + off, fills[f], n ? n - 1 : 0);
#else
				memset_call((char *)buf + 16 + off, fills[f], n);
#endif
				FENCE();
				int a = verify(off, n, fills[f], 1);
				evict();
				int b = verify(off, n, fills[f], 1);
				if (!a || !b)
					printf("MS_MISS:fill=%#x:off=%u:n=%u:fence=%d:evict=%d\n", fills[f], off, n, a, b);
				fence_ok &= a;
				evict_ok &= b;
				++cases;
			}
	check(fence_ok, "fill_after_fence");
	check(evict_ok, "fill_after_eviction");
	check(cases == 2 * 16 * (int)(sizeof lens / sizeof lens[0]), "all_cases_ran");
	printf("MS_DONE:%d:%d:%d\n", checks, failures, cases);
	return failures != 0;
}
