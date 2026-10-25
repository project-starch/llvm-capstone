/* Byte-at-a-time replacements for musl's word-at-a-time string routines.
 *
 * WHY THESE EXIST, AND WHY COMPILING IS NOT ENOUGH. musl scans strings a
 * machine word at a time: read a size_t, test all its bytes for zero, advance.
 * On a flat target reading a few bytes past the terminator is harmless, because
 * the object is inside a page that is mapped anyway. Under capabilities it is
 * not harmless, it is the fault the bounds exist to raise.
 *
 * Measured 2026-09-16, and it is the reason this file was written rather than
 * assumed: a domain calling fprintf halted with
 *
 *   Cap mem access OOB: rs1 = x10, addr = ...da10, size = 8,
 *                       bounds = (...da0f, ...da13)
 *
 * inside __strchrnul, an eight-byte load through a capability four bytes wide
 * over `xdigits` in read-only data. Nothing was wrong with the string; the
 * routine simply read the word containing its last byte.
 *
 * THIS IS THE GAP A COMPILE SURVEY CANNOT SEE. All seven of these files compile
 * cleanly for capstone64 and appear in the archive as successes. They are
 * correct C and they are wrong here, and only running them says so. The port's
 * own survey reports 1355 of 1361 sources compiling, and that number is true
 * and does not mean these seven work.
 *
 * Linked BEFORE libc-capstone.a so these definitions win and musl's are never
 * pulled. That is the same arrangement the SQLite, nginx and PostgreSQL domains
 * already use with beebs_freestanding_string.c, which covers three of these;
 * the set here is the seven musl files that actually read by the word, found by
 * grepping its own ONES/HASZERO/ALIGN idiom rather than by waiting for each to
 * fault in turn.
 *
 * Semantics are musl's, function for function, including the return values that
 * differ between the stp* and str* families.
 */
#include <stddef.h>

size_t strlen(const char *s)
{
	const char *p = s;
	while (*p)
		p++;
	return (size_t)(p - s);
}

char *__strchrnul(const char *s, int c)
{
	c = (char)c;
	for (; *s && *s != c; s++)
		;
	return (char *)s;
}

void *memchr(const void *src, int c, size_t n)
{
	const unsigned char *s = src;
	c = (unsigned char)c;
	for (; n && *s != c; s++, n--)
		;
	return n ? (void *)s : 0;
}

char *stpcpy(char *restrict d, const char *restrict s)
{
	while ((*d = *s)) {
		d++;
		s++;
	}
	return d;
}

char *stpncpy(char *restrict d, const char *restrict s, size_t n)
{
	for (; n && (*d = *s); n--, s++, d++)
		;
	for (size_t i = 0; i < n; i++)
		d[i] = 0;
	return d;
}

size_t strlcpy(char *d, const char *s, size_t n)
{
	size_t len = strlen(s);
	if (n) {
		size_t copy = len < n - 1 ? len : n - 1;
		for (size_t i = 0; i < copy; i++)
			d[i] = s[i];
		d[copy] = 0;
	}
	return len;
}

void *memccpy(void *restrict dest, const void *restrict src, int c, size_t n)
{
	unsigned char *d = dest;
	const unsigned char *s = src;
	c = (unsigned char)c;
	for (; n; n--) {
		if ((*d++ = *s++) == c)
			return d;
	}
	return 0;
}

/* musl reaches several of these through internal hidden aliases, so the aliases
   have to resolve here too or the archive's versions get pulled in for them and
   the word-at-a-time code is back in the image through the side door. */
char *__stpcpy(char *restrict, const char *restrict) __attribute__((alias("stpcpy")));
char *__stpncpy(char *restrict, const char *restrict, size_t) __attribute__((alias("stpncpy")));
/* And the public name of the one routine defined under its internal name: musl's
   strchrnul.o carries both, so a program that calls strchrnul() -- PostgreSQL's
   snprintf.c does, under HAVE_DECL_STRCHRNUL -- pulled that member in beside
   this object, and the link failed on the duplicate __strchrnul. */
char *strchrnul(const char *, int) __attribute__((weak, alias("__strchrnul")));

/* ---- memcpy, memmove, memset: tag-preserving ---------------------------
 *
 * These three are a different problem from the scanning routines above, and a
 * worse one. musl's memcpy copies a machine word at a time within the length
 * it was given, so it never reads out of bounds. What it does instead is copy
 * 8-byte words through a capability that holds 16-byte tagged values, and an
 * 8-byte load-store pair of half a capability yields an untagged one: a struct
 * with a pointer in it, copied by musl's memcpy, comes out with a pointer that
 * faults on first use. Byte-wise copying does the same thing more slowly. The
 * only copy that preserves a tag is a 16-byte capability load and store of a
 * 16-aligned slot, which is what the compiler emits for a pointer-typed access
 * and what this code asks for by copying through `void **`.
 *
 * So: aligned 16-byte slots move as capabilities, and everything else as
 * aligned 8- or 4-byte words where source and target agree in that alignment,
 * bytes otherwise (copy_forward below says why that is the same). A tagged value that is not 16-aligned in both source and target
 * cannot be preserved by any copy, which is also CHERI's rule, and the bytes
 * of it are still copied correctly, only untagged.
 *
 * This is the same rule level0.c's realloc has to follow, and does now, by
 * calling memmove. Found via libc-test's inet_pton, whose inet_ntop compresses
 * the zero run with memmove: the wrong answer there was the visible half of a
 * defect whose invisible half is every pointer-bearing struct any program ever
 * copies.
 */
typedef void *cap_t;
#define CAP_ALIGNED(p) ((((__UINTPTR_TYPE__)(p)) & 15) == 0)

/* CAPSTONE_MEMCPY_PLAIN_GUARD (silicon builds): a granule that holds no capability is copied as two 8-byte words.
 *
 * ISSUES R-29, open on silicon. A 128-bit load of a granule returns a WRONG HIGH HALF whenever a plain store to that
 * granule is still in the write buffer: 0 after a fresh low-word store, the OLD value after a fresh high-word store.
 * The low word and a plain `ld` are always correct. The granule loop below is exactly that load, so on silicon any
 * memcpy of freshly written plain data can lose its high words (llvm-capstone B0 hit it in the delegate runtime,
 * 2026-10-04). Stores through `stc` are not in the hazard's shape.
 *
 * So the copy asks LCC's type query about what the 128-bit load returned: 7 is NOT_CAP. Then it copies the granule
 * with two plain 8-byte loads and stores, which R-29 does not touch. A tagged granule still moves as a capability,
 * so tags survive exactly as before. The type query is TOTAL on the deployed bitstream and on capstone-qemu
 * (op_helper.c, the `imm == 1 && !tag` case). On an older bitstream it RAISES on plain data, which is why this is
 * opt-in per build, as the SQLite silicon build's granule guard (W-12) is, and not a target-wide default. Off, the
 * loop is byte-identical to before. */
#if defined(CAPSTONE_MEMCPY_PLAIN_GUARD) && CAPSTONE_MEMCPY_PLAIN_GUARD
static inline __attribute__((always_inline)) void granule_copy(unsigned char *d, const unsigned char *s)
{
	cap_t v = *(const cap_t *)s;
	unsigned long type;
	__asm__ volatile ("lcc %0, %1, 1" : "=r"(type) : "r"(v));
	if (type == 7) {
		/* Both are 16-aligned here: two `ld` and two `sd`, never a capability (and never bytes, which a
		   memcpy-of-8 from an unsigned char * would compile to). */
		typedef unsigned long __attribute__((__may_alias__, __aligned__(8))) word_t;
		word_t lo = ((const word_t *)s)[0], hi = ((const word_t *)s)[1];
		((word_t *)d)[0] = lo;
		((word_t *)d)[1] = hi;
	} else {
		*(cap_t *)d = v;
	}
}
#define GRANULE_COPY(d, s) granule_copy((d), (s))
#else
#define GRANULE_COPY(d, s) (*(cap_t *)(d) = *(const cap_t *)(s))
#endif

/* Words for what holds no capability. Every scalar store clears the tag of
 * each 16-byte granule it touches, whatever its width, so where the copy
 * cannot preserve a tag (no whole source granule lands on a whole destination
 * granule) aligned words copy the same bytes and clear the same tags as bytes
 * do. Source and destination that agree mod 8 (or mod 4) become aligned
 * together after the same byte head; every word lies wholly inside both
 * ranges, and the pointers stay derived from dst and src. In 538.imagick_r,
 * 47% of the memcpy calls had addresses that differ mod 16 and copied byte by
 * byte, 43M of memcpy's 66M instructions. */
typedef unsigned long __attribute__((__may_alias__, __aligned__(8))) word8_t;
typedef unsigned int __attribute__((__may_alias__, __aligned__(4))) word4_t;
#define WORD_COPY(T, d, s) (*(T *)(d) = *(const T *)(s))

/* Forward copy. Also correct for overlapping ranges with d below s: a word or
 * granule is read before it is written, and the distance between d and s is
 * a multiple of its size whenever it is used. No restrict, so memmove can use
 * it for those. */
static inline __attribute__((always_inline))
void copy_forward(unsigned char *d, const unsigned char *s, size_t n)
{
	__UINTPTR_TYPE__ x = (__UINTPTR_TYPE__)d ^ (__UINTPTR_TYPE__)s;
	if (!(x & 15)) {
		while (n && ((__UINTPTR_TYPE__)d & 7)) { *d++ = *s++; n--; }
		if (n >= 8 && !CAP_ALIGNED(d)) {
			/* the upper half of a granule cannot hold a capability */
			WORD_COPY(word8_t, d, s);
			d += 8; s += 8; n -= 8;
		}
		for (; n >= 16; d += 16, s += 16, n -= 16) GRANULE_COPY(d, s);
		if (n >= 8) {
			/* half a granule: it holds no whole capability either */
			WORD_COPY(word8_t, d, s);
			d += 8; s += 8; n -= 8;
		}
	} else if (!(x & 7)) {
		while (n && ((__UINTPTR_TYPE__)d & 7)) { *d++ = *s++; n--; }
		for (; n >= 8; d += 8, s += 8, n -= 8) WORD_COPY(word8_t, d, s);
	} else if (!(x & 3)) {
		while (n && ((__UINTPTR_TYPE__)d & 3)) { *d++ = *s++; n--; }
		for (; n >= 4; d += 4, s += 4, n -= 4) WORD_COPY(word4_t, d, s);
	}
	while (n--) *d++ = *s++;
}

void *memcpy(void *restrict dst, const void *restrict src, size_t n)
{
	copy_forward(dst, src, n);
	return dst;
}

void *memmove(void *dst, const void *src, size_t n)
{
	unsigned char *d = dst;
	const unsigned char *s = src;
	__UINTPTR_TYPE__ da = (__UINTPTR_TYPE__)d, sa = (__UINTPTR_TYPE__)s;
	if (da == sa || n == 0) return dst;
	if (da < sa || da - sa >= n) {
		/* d below s, or no overlap: forward is safe */
		copy_forward(d, s, n);
		return dst;
	}
	/* d above s and overlapping: backwards, the mirror of copy_forward. */
	__UINTPTR_TYPE__ x = da ^ sa;
	d += n; s += n;
	if (!(x & 15)) {
		while (n && ((__UINTPTR_TYPE__)d & 7)) { *--d = *--s; n--; }
		if (n >= 8 && !CAP_ALIGNED(d)) {
			d -= 8; s -= 8; n -= 8;
			WORD_COPY(word8_t, d, s);
		}
		while (n >= 16) {
			d -= 16; s -= 16; n -= 16;
			GRANULE_COPY(d, s);
		}
		if (n >= 8) {
			d -= 8; s -= 8; n -= 8;
			WORD_COPY(word8_t, d, s);
		}
	} else if (!(x & 7)) {
		while (n && ((__UINTPTR_TYPE__)d & 7)) { *--d = *--s; n--; }
		while (n >= 8) { d -= 8; s -= 8; n -= 8; WORD_COPY(word8_t, d, s); }
	} else if (!(x & 3)) {
		while (n && ((__UINTPTR_TYPE__)d & 3)) { *--d = *--s; n--; }
		while (n >= 4) { d -= 4; s -= 4; n -= 4; WORD_COPY(word4_t, d, s); }
	}
	while (n--) *--d = *--s;
	return dst;
}

void *memset(void *dst, int c, size_t n)
{
	/* A memset cannot create a tag, and overwriting a capability with
	   integer data must clear it. Every scalar store clears the tag of each
	   16-byte granule it touches, whatever its width (capstone-qemu's store
	   hook, cap_mem_map_physical_remove_range), so aligned 8-byte words
	   clear exactly what bytes would, with an eighth of the stores: the
	   byte loop this replaces was a quarter of 519.lbm_r's program
	   instructions. Only whole words inside [dst, dst + n) are
	   written as words; the unaligned edges are bytes. The pointer stays
	   derived from dst, so every store is checked against its bounds. */
	typedef unsigned long __attribute__((__may_alias__, __aligned__(8))) word_t;
	unsigned char *d = dst;
	unsigned long w = (unsigned char)c * 0x0101010101010101ul;
	while (n && ((__UINTPTR_TYPE__)d & 7)) { *d++ = (unsigned char)c; n--; }
	for (; n >= 32; d += 32, n -= 32) {
		word_t *p = (word_t *)d;
		p[0] = w; p[1] = w; p[2] = w; p[3] = w;
	}
	for (; n >= 8; d += 8, n -= 8) *(word_t *)d = w;
	while (n--) *d++ = (unsigned char)c;
	return dst;
}
