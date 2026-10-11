#include "corpus.h"

/* out_string, fix 49f3b0ca9b57. The write buffer is ONE direct malloc from
 * conn_new; neither slabs.c nor cache.c is involved. */

MCH_CASE(5) {
  /* Case 5 -- out_string's CRLF, fix 49f3b0ca9b57. PLAIN HEAP: the guard
   * reserves two bytes for the terminator and the copy writes three.
   *
   * At the fix's parent:
   *
   *     if ((len + 2) > c->wsize) { ... }
   *     memcpy(c->wbuf, str, len);
   *     memcpy(c->wbuf + len, "\r\n", 3);
   *     c->wbytes = len + 2;
   *
   * and the fix:
   *
   *     memcpy(c->wbuf + len, "\r\n", 2);
   *
   * The third byte is the string literal's NUL. At the exact boundary
   * len + 2 == wsize it is written at index wsize -- one past the buffer. Note
   * `c->wbytes = len + 2` already says only two bytes were meant.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long wsize = 16;                 /* a size class: no slack to absorb the crossing */
  const unsigned long len = wsize - 2;            /* the exact boundary the guard admits */
  CHECK(len + 2 <= wsize, 931);                   /* the guard passes, as upstream's does */

  unsigned char *wbuf = calloc((size_t)wsize, 1);
  CHECK(wbuf, 932);
  memset(wbuf, 'x', (size_t)len);

  const unsigned long n = fixed ? 2 : 3;          /* the fix's term */
  unsigned long touched = 0;
  for (unsigned long i = 0; i < n; i++) {
    const unsigned char byte = (i == 0) ? '\r' : (i == 1) ? '\n' : '\0';
    touched = len + i;
    if (touched >= wsize)
      write_probe(wbuf + touched, byte);          /* the labelled crossing */
    else
      wbuf[touched] = byte;
  }

  o->cap = wsize;
  o->touched = touched;
  o->crossed = touched >= wsize;
  o->damage = o->crossed;

  o->defect_text = "the CRLF was copied with length 3, planting the literal's NUL one byte past the "
                   "write buffer at the exact boundary the guard admits";
  o->fixed_text = "the fix copies 2 bytes, which is what the guard reserved and what wbytes claims";
  free(wbuf);
}
