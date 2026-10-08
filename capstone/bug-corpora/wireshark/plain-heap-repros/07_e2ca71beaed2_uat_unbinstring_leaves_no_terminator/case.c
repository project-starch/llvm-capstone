#include "corpus.h"

/* epan/uat.c, uat_unbinstring, fix e2ca71beaed2. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(7) {
  /* Case 7 -- uat_unbinstring's decode buffer, fix e2ca71beaed2. PLAIN HEAP: the
   * buffer is sized exactly for the decoded bytes and then read as a C string.
   *
   * At the fix's parent:
   *
   *     guint len = in_len/2;
   *     buf = g_malloc(len);
   *     *len_p = len;
   *
   * and the fix widens it and zeroes the extra byte:
   *
   *     buf = g_malloc0(len+1);
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long in_len = 32;                /* hex digits */
  const unsigned long len = in_len / 2;           /* 16 decoded bytes: a size class */
  const unsigned long cap = fixed ? len + 1 : len;
  unsigned char *buf = calloc((size_t)cap, 1);
  CHECK(buf, 1231);
  for (unsigned long i = 0; i < len; i++)
    buf[i] = (unsigned char)(0x41 + (i % 26));    /* every decoded byte non-zero */
  CHECK(buf[len - 1] != 0, 1232);                 /* the premise, asserted */

  unsigned long touched = 0;
  unsigned acc = 0;
  for (unsigned long i = 0; ; i++) {
    touched = i;
    if (i >= cap) {
      acc += read_probe(buf + i);                 /* the labelled crossing */
      break;
    }
    if (buf[i] == 0)
      break;
    acc += buf[i];
  }
  (void)acc;

  o->cap = cap;
  o->touched = touched;
  o->crossed = touched >= cap;
  o->extent = 1;
  o->damage = o->crossed;
  o->defect_text = "the decode buffer was sized exactly for its output bytes, so a consumer reading "
                   "it as a C string runs past the allocation";
  o->fixed_text = "the fix allocates len + 1 with g_malloc0, so the extra byte is a terminator";
  free(buf);
}
