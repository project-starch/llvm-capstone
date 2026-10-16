/* __capstone_malloc_remap_usable decides whether a remapped mapping can host representable bounds
 * (runtime/virtual/heap.c). This tests that decision on the host, two-sided: a page-aligned base
 * must be REFUSED once the granule exceeds a page, and a granule-aligned one accepted. Without the
 * refusing cases the fallback could be dead code and nobody would know.
 *
 *   cc -I capstone/runtime/include -o remap-usable-test remap-usable-test.c && ./remap-usable-test
 */
#include <stdio.h>
#include <capstone/representable.h>

/* the decision, as heap.c defines it */
static int usable(unsigned long base, unsigned long bytes)
{
	return capstone_repr_ok(base, capstone_repr_round(bytes ? bytes : 1));
}

static int fails;
static void expect(int want, unsigned long base, unsigned long bytes, const char *what)
{
	int got = usable(base, bytes);
	if (got != want) { printf("FAIL %-52s -> %d, wanted %d\n", what, got, want); fails++; }
	else printf("  %-6s %s\n", got ? "use" : "COPY", what);
}

int main(void)
{
	const unsigned long MiB = 1UL << 20, PAGE = 4096;
	unsigned long aligned = 1UL << 34;              /* aligned to anything we test */

	/* Below 4 MiB the granule is at most a page, so a page-aligned remap is fine. */
	expect(1, aligned + PAGE, 1 * MiB, "1 MiB at a page-aligned base (granule <= page)");
	expect(1, aligned + PAGE, 2 * MiB, "2 MiB at a page-aligned base (granule == page)");

	/* From 4 MiB up the granule exceeds a page: a page-aligned remap must be refused. */
	expect(0, aligned + PAGE,  4 * MiB,  "4 MiB at a page-aligned base");
	expect(0, aligned + PAGE,  8 * MiB,  "8 MiB at a page-aligned base");
	expect(0, aligned + PAGE, 16 * MiB, "16 MiB at a page-aligned base");
	expect(0, aligned + PAGE, 64 * MiB, "64 MiB at a page-aligned base");

	/* and accepted once the base is granule-aligned */
	expect(1, aligned,  4 * MiB,  "4 MiB at a granule-aligned base");
	expect(1, aligned, 64 * MiB, "64 MiB at a granule-aligned base");

	if (fails) { printf("\n%d failure(s)\n", fails); return 1; }
	printf("\nthe fallback fires on a page-aligned base above 4 MiB and not below it\n");
	return 0;
}
