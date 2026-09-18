/* mbsrtowcs without the word-at-a-time scan.
 *
 * The continuation of string_bounds_safe.c for the one musl source outside
 * src/string that reads by the word. That file says its set was found by
 * grepping musl's ONES/HASZERO/ALIGN idiom, and that pattern finds seven
 * files. The idiom that actually matters is `__may_alias__`, and grepping for
 * it finds TWELVE: the ten string routines, calloc, and this one. Eleven were
 * already replaced, by string_bounds_safe.c and by the port's own allocator.
 * This is the twelfth, and it was the last capability fault in libc-test that
 * belonged to this class:
 *
 *   Cap mem access OOB: insn = lw a7, 4(a4), inside mbsrtowcs
 *
 * from libc-test's mbc. The two fast paths below, both guarded by
 * `#ifdef __GNUC__` upstream, load a uint32_t through a `__may_alias__` type
 * and step four bytes at a time:
 *
 *   while (!(( *(w32*)s | *(w32*)s-0x01010101) & 0x80808080)) { s += 4; ... }
 *
 * On flat memory the read that straddles the terminator stays inside the same
 * aligned word and so inside the same page. Under exact bounds it is the fault
 * the bounds exist to raise. Everything else here is musl's code unchanged,
 * including the state-machine resume labels, because the decoder is already
 * byte-at-a-time and there is nothing to improve about it.
 *
 * Linked BEFORE libc-capstone.a so this definition wins; mbsrtowcs.o defines
 * nothing else, so musl's copy is never pulled.
 */
#include <stdint.h>
#include <wchar.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include "internal.h"

size_t mbsrtowcs(wchar_t *restrict ws, const char **restrict src, size_t wn, mbstate_t *restrict st)
{
	const unsigned char *s = (const void *)*src;
	size_t wn0 = wn;
	unsigned c = 0;

	if (st && (c = *(unsigned *)st)) {
		if (ws) {
			*(unsigned *)st = 0;
			goto resume;
		} else {
			goto resume0;
		}
	}

	if (MB_CUR_MAX==1) {
		if (!ws) return strlen((const char *)s);
		for (;;) {
			if (!wn) {
				*src = (const void *)s;
				return wn0;
			}
			if (!*s) break;
			c = *s++;
			*ws++ = CODEUNIT(c);
			wn--;
		}
		*ws = 0;
		*src = 0;
		return wn0-wn;
	}

	if (!ws) for (;;) {
		if (*s-1u < 0x7f) {
			s++;
			wn--;
			continue;
		}
		if (*s-SA > SB-SA) break;
		c = bittab[*s++-SA];
resume0:
		if (OOB(c,*s)) { s--; break; }
		s++;
		if (c&(1U<<25)) {
			if (*s-0x80u >= 0x40) { s-=2; break; }
			s++;
			if (c&(1U<<19)) {
				if (*s-0x80u >= 0x40) { s-=3; break; }
				s++;
			}
		}
		wn--;
		c = 0;
	} else for (;;) {
		if (!wn) {
			*src = (const void *)s;
			return wn0;
		}
		if (*s-1u < 0x7f) {
			*ws++ = *s++;
			wn--;
			continue;
		}
		if (*s-SA > SB-SA) break;
		c = bittab[*s++-SA];
resume:
		if (OOB(c,*s)) { s--; break; }
		c = (c<<6) | *s++-0x80;
		if (c&(1U<<31)) {
			if (*s-0x80u >= 0x40) { s-=2; break; }
			c = (c<<6) | *s++-0x80;
			if (c&(1U<<31)) {
				if (*s-0x80u >= 0x40) { s-=3; break; }
				c = (c<<6) | *s++-0x80;
			}
		}
		*ws++ = c;
		wn--;
		c = 0;
	}

	if (!c && !*s) {
		if (ws) {
			*ws = 0;
			*src = 0;
		}
		return wn0-wn;
	}
	errno = EILSEQ;
	if (ws) *src = (const void *)s;
	return -1;
}
