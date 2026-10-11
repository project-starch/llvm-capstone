#include "corpus.h"

/* epan/to_str.c, oid_to_str_buf, fix 4b15bf76a7f7. The crossed buffer is ONE direct g_malloc-family
 * allocation; no wmem allocator is in the path. */

WSH_CASE(11) {
  /* Case 11 -- oid_to_str_buf's tail reservation, fix 4b15bf76a7f7. PLAIN HEAP:
   * the loop reserved 15 bytes of tail room while the worst case needs 16.
   *
   * At the fix's parent:
   *
   *     if ((bufp - buf) > (buf_len - 15)) {
   *       bufp += g_snprintf(bufp, buf_len-(bufp-buf), ".>>>");
   *       break;
   *     }
   *     ...
   *   *bufp = '\0';
   *
   * and the fix names the worst case:
   *
   *     #define OID_STR_LIMIT (1 + 10 + 4 + 1)  // "." + 10 digits + ".>>>" + '\0'
   *     if ((bufp - buf) > (buf_len - OID_STR_LIMIT)) {
   *
   * g_snprintf returns the WOULD-BE length even when it truncates, so `bufp`
   * advances past the end and the final `*bufp = '\0'` writes there.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long buf_len = 16;               /* a size class */
  const unsigned long reserve = fixed ? 16 : 15;  /* OID_STR_LIMIT vs the old 15 */
  unsigned char *buf = calloc((size_t)buf_len, 1);
  CHECK(buf, 1271);

  /* Enter the tail branch at exactly the cursor the old reservation admits. */
  unsigned long bufp = buf_len - reserve + 11;    /* a maximal subid wrote 11 bytes */
  if (bufp > buf_len) bufp = buf_len;
  /* The truncated ".>>>" advances the cursor by its would-be length, 4. */
  bufp += 4;
  if (bufp > buf_len) bufp = buf_len;

  o->cap = buf_len;
  o->touched = bufp;
  o->crossed = bufp >= buf_len;
  o->extent = 1;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe(buf + bufp, 0);                   /* the labelled crossing: *bufp = '\0' */

  o->defect_text = "the tail reservation was 15 bytes while the worst case needs 16, and g_snprintf "
                   "advances the cursor by its would-be length, so the final terminator lands past "
                   "the buffer";
  o->fixed_text = "the fix reserves OID_STR_LIMIT = 1 + 10 + 4 + 1, the worst case written down";
  free(buf);
}
