/* The module's arena test, after it became the representability predicate (capstone_vm.c add(),
 * the exact-bounds-off branch). This is a HOST test of the same shared header the module compiles,
 * because the module itself cannot be built without a prepared kernel tree.
 *
 * Two-sided, and the negative half is the point: it feeds the predicate the four shapes
 * grant-repr-test guards against, so a predicate that stopped refusing them fails here rather than
 * admitting a widened arena on silicon. It also asserts the OLD rule's verdicts where the two
 * differ, so the loosening is exactly the intended one and not wider.
 *
 *   cc -I capstone/runtime/include -o module-arena-test module-arena-test.c && ./module-arena-test
 */
#include <stdio.h>
#include <capstone/representable.h>

/* The rule the module enforced BEFORE: a naturally aligned power of two. */
static int old_rule(unsigned long base, unsigned long len)
{
	if (!len || (len & (len - 1))) return 0;
	return (base & (len - 1)) == 0;
}
static int fails;
static void expect(int want, unsigned long base, unsigned long len, const char *what)
{
	int got = capstone_repr_ok(base, len);
	if (got != want) {
		printf("FAIL  want=%d got=%d  base=0x%lx len=%lu  %s\n", want, got, base, len, what);
		fails++;
	}
}

int main(void)
{
	unsigned long b, n;

	/* NEGATIVE: the four shapes grant-repr-test guards. These must STAY refused; the whole
	 * point of keeping a refusal is that QEMU's encoder would silently widen them instead. */
	expect(0, 0x40000000UL + 4096, 8u << 20,       "non-pow2 length at a page-aligned base");
	expect(0, 0x40000000UL + 4096, (8u << 20) + 8, "length not a granule multiple");
	expect(0, 0x40000001UL, 4096,                  "base one byte off (the edge=1 descriptor)");
	expect(0, 0x40000010UL, 1u << 20,              "static arena at a 16-byte base, 1 MiB long");

	/* POSITIVE: the launcher's own arena shapes still pass. */
	for (b = 4096; b <= (1UL << 22); b <<= 1)
		expect(1, 0x40000000UL & ~(b - 1), b, "power-of-two arena at a naturally aligned base");

	/* THE LOOSENING, stated as cases rather than as a claim: shapes the old rule refused and the
	 * encoder represents exactly. Each is a real mallocng mapping size. */
	{
		static const unsigned long real[] = {20480, 28672, 36864, 49152, 122880,
						     135168, 200704, 1126400, 6307840};
		unsigned i;
		for (i = 0; i < sizeof real / sizeof *real; ++i) {
			n = real[i];
			b = 0x40000000UL;           /* 4096-aligned, as the launcher provides */
			if (b % capstone_repr_granule(n)) {
				printf("FAIL  test setup: 0x%lx is not granule-aligned for %lu\n", b, n);
				fails++;
				continue;
			}
			expect(1, b, n, "real mallocng mapping, granule-aligned base");
			if (old_rule(b, n)) {
				printf("FAIL  the old rule already admitted %lu -- not a loosening\n", n);
				fails++;
			}
		}
	}

	/* AND the loosening is bounded: the base must still be a granule multiple, so the alignment
	 * half is not simply dropped. It only bites where the granule EXCEEDS a page, i.e. at and
	 * above 4 MiB -- a first version of this test asserted that 8192 at 0x40001000 is refused,
	 * which is wrong: granule(8192) is 16, so every page-aligned base satisfies it. That is
	 * precisely why the relaxation removes the boot-to-boot nondeterminism, and the test now
	 * says so with a case instead of a claim. */
	expect(1, 0x40000000UL + 4096, 8192,  "8192 at any page-aligned base: granule is only 16");
	expect(0, 0x40000000UL + 4096, 8u << 20,
	       "8 MiB at a page-aligned base: granule 16384, so the base is NOT a multiple");
	expect(1, 0x40000000UL + 16384, 8u << 20,
	       "8 MiB at a granule-aligned base");
	/* The LENGTH half, separated: an ALIGNED base whose length is not a granule multiple. Without
	 * this case a predicate that checks only the base passes the whole test -- which it did, and
	 * is why the case is here rather than in a comment. */
	expect(0, 0x40000000UL, (8u << 20) + 8,
	       "aligned base, length not a granule multiple");

	if (fails) {
		printf("module-arena-test: %d FAILURES\n", fails);
		return 1;
	}
	printf("module-arena-test: the four guarded shapes stay refused, the launcher's shapes pass,\n"
	       "  nine real mallocng mappings are newly admitted, and the alignment half still binds\n");
	return 0;
}
