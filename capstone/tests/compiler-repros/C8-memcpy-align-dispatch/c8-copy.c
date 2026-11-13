/* C8 directed test: constant-size copies whose 16-byte alignment the compiler cannot prove (C-81), under
 * -capstone-memcpy-align-dispatch. Every case runs the copy under test and the runtime's memcpy (called through a
 * volatile pointer, so it stays a call) on identically prepared buffers, and requires the same bytes and the same tag
 * in every granule; it also requires the expected tags outright:
 *   both cursors 16-aligned   the copied granules keep their tags and name the same object;
 *   dst 8 mod 16              no granule moves as a capability, every granule the copy touches loses its tag;
 *   src 8 mod 16              the same;
 *   both 8 mod 16             the same at 16 bytes; at 32 the middle granule is whole on both sides and keeps its tag.
 * Also an exact self-copy (d == s) at both alignments, matched against memcpy's own self-copy.
 * Sizes 16 (the pass's scope) and 32 (left a memcpy call by the pass; the expectation covers it so a later extension
 * is tested by the same file). The control is a plain-word copy of the aligned case, which must lose the tag: it shows the tag
 * read can report a loss. Tags are read with the compiler builtin, never inferred from values. */
#include <stdio.h>
#include <string.h>

#define TAG(p) __builtin_capstone_cap_get_tag((void *)(p))
#define BARRIER() __asm__ volatile("" ::: "memory")
#define G 8 /* granules per buffer */

static int cells[3 * G];
static void *src[G] __attribute__((aligned(16)));
static void *dst[G] __attribute__((aligned(16)));
static void *ref[G] __attribute__((aligned(16)));
static int checks, failures;

static void check(int ok, const char *name)
{
	++checks;
	failures += !ok;
	printf("C8_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
}

/* The copies under test: `unsigned long *` tells the compiler 8, never 16. */
#ifndef C8_MUTANT
__attribute__((noinline)) void copy16(unsigned long *d, const unsigned long *s) { memcpy(d, s, 16); }
#else
/* The mutation control: what a dispatch whose capability arm were wrong would do (words always). Built with
   -DC8_MUTANT, the both_aligned_16 checks must FAIL through the same harness. */
__attribute__((noinline)) void copy16(unsigned long *d, const unsigned long *s)
{
	volatile unsigned long *vd = d;
	const volatile unsigned long *vs = s;
	vd[0] = vs[0];
	vd[1] = vs[1];
}
#endif
__attribute__((noinline)) void copy32(unsigned long *d, const unsigned long *s) { memcpy(d, s, 32); }
/* The control: plain 8-byte words, which can never carry a tag. */
__attribute__((noinline)) void copy16_words(unsigned long *d, const unsigned long *s)
{
	volatile unsigned long *vd = d;
	const volatile unsigned long *vs = s;
	vd[0] = vs[0];
	vd[1] = vs[1];
}
static void *(*volatile memcpy_call)(void *, const void *, unsigned long) = memcpy;

static void fill(void)
{
	for (int g = 0; g < G; ++g) {
		src[g] = &cells[g];
		dst[g] = &cells[G + g];
		ref[g] = &cells[G + g];
	}
	BARRIER();
}

/* Returns whether dst equals ref byte for byte and tag for tag. */
static int same_as_memcpy(void)
{
	const unsigned char *a = (const unsigned char *)dst, *b = (const unsigned char *)ref;
	for (unsigned i = 0; i < sizeof dst; ++i)
		if (a[i] != b[i])
			return 0;
	for (int g = 0; g < G; ++g)
		if (TAG(dst[g]) != TAG(ref[g]))
			return 0;
	return 1;
}

static void run_case(const char *name, unsigned doff, unsigned soff, unsigned n)
{
	char nm[96];
	fill();
	unsigned long *d = (unsigned long *)((char *)dst + 16 + doff);
	const unsigned long *s = (const unsigned long *)((char *)src + 16 + soff);
	if (n == 16)
		copy16(d, s);
	else
		copy32(d, s);
	memcpy_call((char *)ref + 16 + doff, s, n);
	BARRIER();
	snprintf(nm, sizeof nm, "%s_%u_matches_memcpy", name, n);
	check(same_as_memcpy(), nm);
	/* The expected tag of each destination granule, from the geometry (memcpy's rule): a granule wholly inside the
	   copied range keeps a capability iff the source bytes that land on it are a whole source granule, i.e. the two
	   offsets agree mod 16; a granule the copy only partly covers loses its tag; the others are untouched. */
	int ok = 1;
	unsigned lo = 16 + doff, hi = 16 + doff + n;
	for (unsigned g = 0; g < G; ++g) {
		unsigned a = 16 * g, b = a + 16;
		if (b <= lo || a >= hi) {
			ok &= TAG(dst[g]) == 1 && dst[g] == &cells[G + g];
		} else if (a >= lo && b <= hi && doff == soff) {
			ok &= TAG(dst[g]) == 1 && dst[g] == &cells[g] && *(int *)dst[g] == (int)g * 7 + 1;
		} else {
			ok &= TAG(dst[g]) == 0;
		}
	}
	snprintf(nm, sizeof nm, "%s_%u_tags", name, n);
	check(ok, nm);
}

/* An exact self-copy (d == s, which llvm.memcpy permits and an aggregate self-assignment lowers to): every load
 * precedes every store, so the result is memcpy's own self-copy. Aligned, the tag survives; at 8 mod 16 the word path
 * rewrites the bytes and, as memcpy does, clears the tags of the two granules it touches. */
static void self_case(const char *name, unsigned off)
{
	char nm[96];
	fill();
	unsigned long *p = (unsigned long *)((char *)dst + 16 + off);
	copy16(p, p);
	memcpy_call((char *)ref + 16 + off, (char *)ref + 16 + off, 16);
	BARRIER();
	snprintf(nm, sizeof nm, "%s_matches_memcpy", name);
	check(same_as_memcpy(), nm);
	int ok = 1;
	for (unsigned g = 0; g < G; ++g) {
		int touched = off ? (g == 1 || g == 2) : g == 1;
		ok &= TAG(dst[g]) == (touched ? !off : 1) && (!touched || off || dst[g] == &cells[G + g]);
	}
	snprintf(nm, sizeof nm, "%s_tags", name);
	check(ok, nm);
}

int main(void)
{
	for (int i = 0; i < 3 * G; ++i)
		cells[i] = i * 7 + 1;
	static const struct { const char *name; unsigned doff, soff; } cases[] = {
		{"both_aligned", 0, 0}, {"dst_misaligned", 8, 0}, {"src_misaligned", 0, 8}, {"both_8_mod_16", 8, 8}};
	for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; ++c) {
		run_case(cases[c].name, cases[c].doff, cases[c].soff, 16);
		run_case(cases[c].name, cases[c].doff, cases[c].soff, 32);
	}
	self_case("self_aligned", 0);
	self_case("self_8_mod_16", 8);
	fill();
	copy16_words((unsigned long *)((char *)dst + 16), (const unsigned long *)((char *)src + 16));
	BARRIER();
	check(TAG(dst[1]) == 0, "control_word_copy_drops_tag");
	printf("C8_DONE:%d:%d\n", checks, failures);
	return failures != 0;
}
