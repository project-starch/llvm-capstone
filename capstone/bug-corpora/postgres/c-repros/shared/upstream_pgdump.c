/* Verbatim from PostgreSQL 17.5, src/bin/pg_dump/common.c:1080-1117.
 * Nothing in the body is changed. The two lines the case turns on are
 * :1097  `if (argNum >= arraysize) pg_fatal(...)`   -- rejects MORE than
 *        arraysize, so exactly arraysize passes, and
 * :1115  `while (argNum < arraysize) array[argNum++] = InvalidOid;` -- the
 *        zero padding, which runs zero times when the array came back full.
 * Together they mean a list of exactly arraysize oids leaves no terminator.
 * atooid and pg_fatal are pg_dump's; they are given their upstream meaning
 * here (an unsigned-long parse, and an exit that is never a verdict).
 */
#include "corpus.h"
#include <stdlib.h>
#include <stdio.h>

/* THE ONE DEVIATION FROM UPSTREAM IN THIS FILE, and why it is here.
 *
 * parseOidArray calls isdigit(). On this guest, a STATIC purecap binary
 * SIGPROTs inside isdigit() before main gets anywhere -- measured with a
 * three-line program: `isdigit('1')` traps whether or not setlocale(LC_ALL,
 * "C") ran first, and setlocale itself succeeds and returns "C". Dynamic
 * linking is not an escape: the rtld refuses the binary outright with
 * "Traditional TLS not supported", and the symbol it names is
 * _ThreadRuneLocale -- the same locale machinery, reached the same way.
 *
 * So the choice is between not running this case on this arm at all and
 * substituting the test. The substitution is provably equivalent for this
 * input: in the C locale -- which is what setlocale reports -- isdigit(c) is
 * exactly c >= '0' && c <= '9', and the only other character parseOidArray
 * accepts is '-', which it tests separately. Nothing about the defect lives
 * here: the defect is the `>= arraysize` guard and the padding loop below,
 * both untouched.
 *
 * Recorded in the case's `fidelity` field as well, because a verbatim copy
 * that is not quite verbatim is exactly the kind of thing that should not be
 * discoverable only by reading the source. */
#define isdigit(c) ((c) >= '0' && (c) <= '9')

#define atooid(x) ((Oid) strtoul((x), NULL, 10))

static void pg_fatal_parse(const char *what, const char *str) {
  /* pg_dump exits here. For the corpus that is a CONTROL failure, not a
   * verdict: it means the input never reached the defect. */
  fprintf(stderr, "CONTROL-FAILED pg_fatal: %s \"%s\"\n", what, str);
  exit(75);
}

void parseOidArray(const char *str, Oid *array, int arraysize)
{
	int			j,
				argNum;
	char		temp[100];
	char		s;

	argNum = 0;
	j = 0;
	for (;;)
	{
		s = *str++;
		if (s == ' ' || s == '\0')
		{
			if (j > 0)
			{
				if (argNum >= arraysize)
					pg_fatal_parse("could not parse numeric array: too many numbers", str);
				temp[j] = '\0';
				array[argNum++] = atooid(temp);
				j = 0;
			}
			if (s == '\0')
				break;
		}
		else
		{
			if (!(isdigit((unsigned char) s) || s == '-') ||
				j >= sizeof(temp) - 1)
				pg_fatal_parse("could not parse numeric array: invalid character in number", str);
			temp[j++] = s;
		}
	}

	while (argNum < arraysize)
		array[argNum++] = InvalidOid;
}
