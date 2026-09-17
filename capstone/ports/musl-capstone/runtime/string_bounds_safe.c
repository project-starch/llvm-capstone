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
