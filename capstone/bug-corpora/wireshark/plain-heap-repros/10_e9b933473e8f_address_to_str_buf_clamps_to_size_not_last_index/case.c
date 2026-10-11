#include "corpus.h"

/* epan/to_str.c, address_to_str_buf's AT_URI arm, fix e9b933473e8f. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(10) {
  /* Case 10 -- address_to_str_buf's AT_URI arm, fix e9b933473e8f. PLAIN HEAP:
   * the copy length is clamped to buf_len, and the terminator is then stored at
   * buf[copy_len].
   *
   * At the fix's parent:
   *
   *     case AT_URI: {
   *       int copy_len = addr->len < buf_len ? addr->len : buf_len;
   *       memmove(buf, addr->data, copy_len );
   *       buf[copy_len] = '\0';
   *       }
   *
   * and the fix clamps to one less:
   *
   *       int copy_len = addr->len < (buf_len - 1) ? addr->len : (buf_len - 1);
   *
   * The memmove is in bounds; the terminator is not.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long buf_len = 16;               /* a size class */
  const unsigned long addr_len = 32;              /* longer than the buffer */
  CHECK(addr_len >= buf_len, 1261);               /* the premise, asserted */

  unsigned char *buf = calloc((size_t)buf_len, 1);
  CHECK(buf, 1262);

  const unsigned long copy_len = fixed
      ? (addr_len < buf_len - 1 ? addr_len : buf_len - 1)
      : (addr_len < buf_len ? addr_len : buf_len);
  for (unsigned long i = 0; i < copy_len; i++)
    buf[i] = (unsigned char)'u';                  /* the memmove, in bounds either way */

  o->cap = buf_len;
  o->touched = copy_len;
  o->crossed = copy_len >= buf_len;
  o->extent = 1;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe(buf + copy_len, 0);               /* the labelled crossing: buf[copy_len] = '\0' */

  o->defect_text = "the copy length was clamped to buf_len rather than buf_len - 1, so the "
                   "terminator stored at buf[copy_len] lands one byte past the buffer";
  o->fixed_text = "the fix clamps to buf_len - 1, leaving room for the terminator";
  free(buf);
}
