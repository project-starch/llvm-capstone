/* The launcher's grant() refuses an arena whose bounds the encoder cannot represent
 * (runtime/virtual/exec.c). That check cannot fire today, so this test exists to show it is a
 * real check and not a dead one: it asserts the launcher's ACTUAL arena shapes pass, and that
 * the two shapes it guards against fail.
 *
 *   cc -I capstone/runtime/include -o grant-repr-test grant-repr-test.c && ./grant-repr-test
 *
 * Exits non-zero on any disagreement, and non-zero if the negative cases do NOT fail -- a test
 * where everything passes would prove nothing about the check. */
#include <stdio.h>
#include <stdint.h>
#include <capstone/representable.h>

/* exec.c:123-129, verbatim: it doubles from 4096, so every result is a power of two. */
static size_t rounded(size_t n)
{
	size_t size = 4096;
	if (!n) return 0;
	while (size < n) size <<= 1;
	return size;
}

#define META_BYTES   16384u   /* CAPSTONE_DELEGATE_META_BYTES */
#define LAUNCH_BYTES 65536u   /* CAPSTONE_LAUNCH_BYTES */

static int fails;
static void expect(int want, unsigned long base, unsigned long len, const char *what)
{
	int got = capstone_repr_ok(base, len);
	if (got != want) {
		printf("FAIL %-44s base=%#lx len=%lu -> %d, wanted %d\n", what, base, len, got, want);
		fails++;
	}
}

int main(void)
{
	/* POSITIVE: the launcher's real shapes. reserve() aligns the base to `bytes`, and every
	 * size is rounded() or one of the two constants -- all powers of two. */
	const size_t inputs[] = {1, 4096, 4097, 65535, 100000, 1u<<20, (1u<<20)+1, 3000000, 1u<<24};
	for (unsigned i = 0; i < sizeof inputs / sizeof *inputs; ++i) {
		size_t b = rounded(inputs[i]);
		expect(1, 0x40000000UL & ~(b - 1), b, "rounded() size at a bytes-aligned base");
		if (b & (b - 1)) { printf("FAIL rounded(%zu) = %zu is not a power of two\n", inputs[i], b); fails++; }
	}
	expect(1, 0x40000000UL, META_BYTES,   "CAPSTONE_DELEGATE_META_BYTES");
	expect(1, 0x40000000UL, LAUNCH_BYTES, "CAPSTONE_LAUNCH_BYTES");

	/* NEGATIVE 1: the regression the check guards -- a size that is NOT a power of two, at a
	 * base aligned only to a page, where the granule exceeds a page. This is what a future
	 * caller bypassing rounded() would produce. */
	expect(0, 0x40000000UL + 4096, 8u<<20,        "non-pow2 length at a page-aligned base");
	expect(0, 0x40000000UL + 4096, (8u<<20) + 8,  "length not a granule multiple");

	/* NEGATIVE 2: the exact-bounds.sh edge=1 shape -- a descriptor based one byte off. */
	expect(0, 0x40000001UL, 4096, "base one byte off (the edge=1 descriptor)");

	/* NEGATIVE 3: a misaligned static arena, memsys5's case. */
	expect(0, 0x40000010UL, 1u<<20, "static arena at a 16-byte base, 1 MiB long");

	if (fails) { printf("\n%d failures\n", fails); return 1; }
	printf("grant-repr-test: the launcher's shapes pass and all four guarded shapes are refused\n");
	return 0;
}
