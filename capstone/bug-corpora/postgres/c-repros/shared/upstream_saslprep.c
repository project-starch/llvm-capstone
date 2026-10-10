/* Verbatim from PostgreSQL 17.5:
 *   pg_utf8_string_len  src/common/saslprep.c:1006-1025  (the defect)
 *   pg_utf_mblen        src/common/wchar.c               (returns the DECLARED length)
 *   pg_utf8_islegal     src/common/wchar.c               (reads that many bytes)
 * Both helpers are `static`/internal upstream and are exported here only so the
 * case can link them. No line of logic is changed. The defect is that
 * saslprep.c:1013's `while (*p)` is evaluated at the LEAD byte, so a sequence
 * whose declared length runs past the terminator is still handed to islegal.
 */
#include "corpus.h"
#include <stdbool.h>

static int pg_utf_mblen(const unsigned char *s)
{
	int			len;

	if ((*s & 0x80) == 0)
		len = 1;
	else if ((*s & 0xe0) == 0xc0)
		len = 2;
	else if ((*s & 0xf0) == 0xe0)
		len = 3;
	else if ((*s & 0xf8) == 0xf0)
		len = 4;
	else
		len = 1;
	return len;
}

static bool pg_utf8_islegal(const unsigned char *source, int length)
{
	unsigned char a;

	switch (length)
	{
		default:
			/* reject lengths 5 and 6 for now */
			return false;
		case 4:
			a = source[3];
			if (a < 0x80 || a > 0xBF)
				return false;
			/* FALL THRU */
		case 3:
			a = source[2];
			if (a < 0x80 || a > 0xBF)
				return false;
			/* FALL THRU */
		case 2:
			a = source[1];
			switch (*source)
			{
				case 0xE0:
					if (a < 0xA0 || a > 0xBF)
						return false;
					break;
				case 0xED:
					if (a < 0x80 || a > 0x9F)
						return false;
					break;
				case 0xF0:
					if (a < 0x90 || a > 0xBF)
						return false;
					break;
				case 0xF4:
					if (a < 0x80 || a > 0x8F)
						return false;
					break;
				default:
					if (a < 0x80 || a > 0xBF)
						return false;
					break;
			}
			/* FALL THRU */
		case 1:
			a = *source;
			if (a >= 0x80 && a < 0xC2)
				return false;
			if (a > 0xF4)
				return false;
			break;
	}
	return true;
}

int pg_utf8_string_len(const char *source)
{
	const unsigned char *p = (const unsigned char *) source;
	int			l;
	int			num_chars = 0;

	while (*p)
	{
		l = pg_utf_mblen(p);

		if (!pg_utf8_islegal(p, l))
			return -1;

		p += l;
		num_chars++;
	}

	return num_chars;
}
