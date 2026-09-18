/* __fputwc_unlocked without arithmetic on a null pointer.
 *
 * musl asks whether the stream's write buffer has room for one more multibyte
 * character by moving the write cursor forward and comparing:
 *
 *   } else if (f->wpos + MB_LEN_MAX < f->wend) {
 *
 * Before __towrite has run, wpos and wend are both null, and `wpos + 4` is
 * then arithmetic on a null pointer. C says that is undefined and every flat
 * libc says it is four; a capability machine says it is a trap. The RTL raises
 * here, not only the emulator: capstone-ariane/core/anvil_build/
 * capstone_flu_unit.anvil:57-60 raises UNEXPECTED_OPERAND when CINCOFFSETIMM
 * gets a NOT_CAP operand, and QEMU reports the same as cause 24 with val=0x0.
 * libc-test's swprintf died on it, at `cincoffsetimm a2, a0, 0x4`.
 *
 * The question the line is asking is how much room is left, so ask that. The
 * two cursors go through uintptr_t, which is 64 bits here, so the subtraction
 * is integer arithmetic that cannot trap and cannot be a capability. With both
 * pointers null the answer is zero and the slow path runs, which is what
 * happens on flat memory too. Nothing else in this file differs from musl's.
 *
 * WORTH KNOWING BEFORE COPYING THIS PATTERN AROUND: across the whole of
 * libc-test, 49 tests run with CAPSTONE_CINC_UNTAGGED_SURVIVE=1, this is the
 * ONLY site in musl that does it. The class is one function, not a front. That
 * measurement is what says a one-line change here is enough and an ISA
 * decision about permissive capability arithmetic, while a real question, is
 * not what this port is waiting for.
 *
 * fputwc.o also defines fputwc and the two weak aliases, so all four have to
 * be here or the linker pulls musl's object back in for them and the
 * definitions collide.
 */
#include "stdio_impl.h"
#include "locale_impl.h"
#include <wchar.h>
#include <limits.h>
#include <ctype.h>
#include <stdint.h>

wint_t __fputwc_unlocked(wchar_t c, FILE *f)
{
	char mbc[MB_LEN_MAX];
	int l;
	locale_t *ploc = &CURRENT_LOCALE, loc = *ploc;

	if (f->mode <= 0) fwide(f, 1);
	*ploc = f->locale;

	if (isascii(c)) {
		c = putc_unlocked(c, f);
	} else if ((uintptr_t)f->wend - (uintptr_t)f->wpos > MB_LEN_MAX) {
		l = wctomb((void *)f->wpos, c);
		if (l < 0) c = WEOF;
		else f->wpos += l;
	} else {
		l = wctomb(mbc, c);
		if (l < 0 || __fwritex((void *)mbc, l, f) < l) c = WEOF;
	}
	if (c==WEOF) f->flags |= F_ERR;
	*ploc = loc;
	return c;
}

wint_t fputwc(wchar_t c, FILE *f)
{
	FLOCK(f);
	c = __fputwc_unlocked(c, f);
	FUNLOCK(f);
	return c;
}

weak_alias(__fputwc_unlocked, fputwc_unlocked);
weak_alias(__fputwc_unlocked, putwc_unlocked);
