#include "corpus.h"

/* librist_read's copy into the caller's buffer, fix 8880a174d0. The destination
 * is ONE direct allocation; here it is the platform's calloc. */

FFH_CASE(7) {
  /* Case 7 -- librist_read, fix 8880a174d0. PLAIN HEAP: the copy length is
   * taken from the SOURCE payload and the destination's own size is discarded.
   *
   * At the fix's parent:
   *
   *     size = data_block->payload_len;
   *
   * and the fix:
   *
   *     size = FFMIN(data_block->payload_len, size);
   *
   * `size` arrives holding the caller's buffer size. Overwriting it with the
   * payload's length is the defect: a payload larger than the buffer is then
   * copied in full.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long buf_size = 32;       /* what the caller provides */
  const long payload_len = 48;    /* what the peer sent */
  CHECK(payload_len > buf_size, 771);   /* the claim, asserted */

  unsigned char *dst = calloc((size_t)buf_size, 1);
  CHECK(dst, 772);
  unsigned char *payload = calloc((size_t)payload_len, 1);
  CHECK(payload, 773);
  for (long i = 0; i < payload_len; i++)
    payload[i] = (unsigned char)(0x30 + i);

  const long n = fixed ? (payload_len < buf_size ? payload_len : buf_size) : payload_len;
  long touched = -1;
  for (long i = 0; i < n && i < buf_size; i++) {
    touched = i;
    dst[i] = payload[i];
  }

  /* ONE byte past, not the whole overrun. A write loop that really ran the full
   * extent corrupts the next chunk header and glibc aborts in free() -- rc 134,
   * which is an infrastructure failure, not a verdict. The crossing is what the
   * row measures; `extent` records how far the unreduced write would run. This
   * is the same shape case 3 uses. */
  if (n > buf_size) {
    touched = buf_size;
    write_probe_u8(&dst[buf_size], payload[buf_size]);   /* the labelled crossing */
  }

  o->cap = (unsigned long)buf_size;
  o->touched = touched;
  o->crossed = touched >= buf_size;
  o->extent = payload_len - buf_size;
  o->damage = o->crossed;

  o->defect_text = "the read overwrote the caller's buffer size with the payload's length, so a "
                   "payload larger than the buffer was copied past its end";
  o->fixed_text = "the fix takes FFMIN of the payload length and the caller's size, so the copy fits";
  free(payload);
  free(dst);
}
