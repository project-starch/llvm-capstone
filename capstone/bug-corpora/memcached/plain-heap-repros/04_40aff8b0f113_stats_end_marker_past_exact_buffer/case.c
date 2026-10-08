#include "corpus.h"

/* process_stat's bare "stats" path, fix 40aff8b0f113. The response buffer is
 * ONE direct malloc; neither slabs.c nor cache.c is involved. */

MCH_CASE(4) {
  /* Case 4 -- the stats response buffer, fix 40aff8b0f113. PLAIN HEAP: the
   * buffer was sized for exactly the two stat blobs, and then the "END\r\n"
   * trailer was written at the position just past them.
   *
   * At the fix's parent:
   *
   *     buf = malloc(server_statlen + engine_statlen);
   *     ptr = buf;
   *     memcpy(ptr, server_statbuf, server_statlen);
   *     ptr += server_statlen;
   *     memcpy(ptr, engine_statbuf, engine_statlen);
   *     ptr += engine_statlen;
   *     engine_statlen += append_ascii_stats(ptr, NULL, 0, NULL, 0);
   *
   * and the fix widens the allocation, by SIX and not five because the sprintf
   * inside append_ascii_stats also plants a NUL:
   *
   *     /* 6 is: strlen("END\r\n") + strlen("\0") *\/
   *     buf = calloc(1, server_statlen + engine_statlen + 6);
   *
   * After both copies `ptr` sits exactly at the end of the allocation, so every
   * byte of the trailer is outside it.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long server_statlen = 48;
  const unsigned long engine_statlen = 16;
  const unsigned long trailer = 6;                    /* "END\r\n" plus its NUL */
  const unsigned long payload = server_statlen + engine_statlen;      /* 64 */
  const unsigned long bytes = fixed ? payload + trailer : payload;
  CHECK(payload % 16 == 0, 921);  /* the buggy arm lands on a size class: no slack */

  unsigned char *buf = calloc((size_t)bytes, 1);
  CHECK(buf, 922);
  memset(buf, 'S', (size_t)server_statlen);
  memset(buf + server_statlen, 'E', (size_t)engine_statlen);

  /* The trailer is written at `ptr`, which is `payload` bytes in. The probe
   * touches the first of its six bytes; all six lie outside on the buggy arm. */
  o->cap = bytes;
  o->touched = payload;
  o->crossed = payload >= bytes;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe(buf + payload, (unsigned char)'E');   /* the labelled crossing */

  o->defect_text = "the buffer was sized as the sum of the two stat blobs, so the six-byte "
                   "END\\r\\n trailer written just past them lies entirely outside it";
  o->fixed_text = "the fix allocates the payload plus 6 -- five for END\\r\\n and one for the NUL "
                  "sprintf plants -- so the trailer fits";
  free(buf);
}
