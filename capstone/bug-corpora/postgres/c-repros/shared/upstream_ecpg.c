/* Verbatim from PostgreSQL 17.5, src/interfaces/ecpg/ecpglib/data.c:139-188.
 * Only the name of the entry point changes -- `hex_decode` there is `static`,
 * so it is exported here as `ecpg_hex_decode` to be callable from a case. The
 * body, the types and above all the `unsigned len` parameter are untouched:
 * that parameter is the defect (CVE-2026-16241). data.c:532 passes it a `long`
 * that data.c:530-531 can leave negative, and the conversion makes it enormous.
 */
#include "corpus.h"

typedef signed char int8;

static inline char get_hex(char c) {
	static const int8 hexlookup[128] = {
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, -1, -1,
		-1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	};
	int			res = -1;

	if (c > 0 && c < 127)
		res = hexlookup[(unsigned char) c];

	return (char) res;
}

unsigned ecpg_hex_decode(const char *src, unsigned len, char *dst)
{
	const char *s,
			   *srcend;
	char		v1,
				v2,
			   *p;

	srcend = src + len;
	s = src;
	p = dst;
	while (s < srcend)
	{
		if (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r')
		{
			s++;
			continue;
		}
		v1 = get_hex(*s++) << 4;
		if (s >= srcend)
			return -1;

		v2 = get_hex(*s++);
		*p++ = v1 | v2;
	}

	return p - dst;
}
