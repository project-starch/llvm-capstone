#include "corpus.h"

/* ws_strptime's timezone parse, fix c556b648aa. The string is ONE direct
 * allocation; wsutil does not use wmem. */

WSH_CASE(3) {
  /* Case 3 -- ws_strptime timezone parse, fix c556b648aa. PLAIN HEAP: the scan
   * over the timezone name had no case for the string terminator, so on an
   * empty timezone it consumed the NUL like any other character and stepped on
   * to the byte after it. The fix adds the missing arm:
   *
   *     case '\0':
   *             goto out;
   *
   * The bound here is a TERMINATOR, not a length, which is why the fix is a new
   * switch arm rather than an arithmetic change. The allocation holds exactly
   * the empty string -- one byte, the NUL -- so the byte after it is past the
   * end.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long len = 1;                  /* just the NUL: an empty timezone */
  unsigned char *tz = calloc((size_t)len, 1);
  CHECK(tz, 821);
  tz[0] = 0;                                    /* the terminator */
  CHECK(tz[0] == 0, 822);                       /* the claim, asserted */

  unsigned long touched = 0;
  unsigned acc = 0;
  unsigned long i = 0;
  for (;;) {
    unsigned char c = tz[i];
    if (c == 0 && fixed)
      break;                                    /* the fix's `case '\0': goto out;` */
    /* pre-fix: the NUL falls through to the default arm and the scan steps on */
    i++;
    touched = i;
    if (i >= len) {
      acc += read_probe(&tz[i]);                /* the labelled crossing */
      break;                                    /* the reduction's own stop */
    }
    acc += c;
  }
  (void)acc;

  o->cap = len;
  o->touched = touched;
  o->crossed = touched >= len;
  o->extent = 1;
  o->damage = o->crossed;

  o->defect_text = "the timezone scan had no case for the terminator, so it consumed the NUL and read "
                   "the byte after it, one past the allocation";
  o->fixed_text = "the fix's `case '\\0': goto out;` stops the scan at the terminator";
  free(tz);
}
