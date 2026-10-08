#include "corpus.h"

/* mpegenc's system-header writer, fix b2df2f4f22. The output buffer is ONE
 * direct allocation; here it is the platform's calloc. */

FFH_CASE(8) {
  /* Case 8 -- mpegenc put_system_header, fix b2df2f4f22. PLAIN HEAP: the bit
   * writer was initialised with a CONSTANT capacity of 128 regardless of how
   * much of the buffer was left at the cursor.
   *
   * At the fix's parent:
   *
   *     static int put_system_header(AVFormatContext *ctx, uint8_t *buf, int id)
   *     ...
   *         init_put_bits(&pb, buf, 128);
   *
   * and the fix threads the real remaining size through:
   *
   *     static int put_system_header(AVFormatContext *ctx, uint8_t *buf, int buf_size, int id)
   *     ...
   *         init_put_bits(&pb, buf, buf_size);
   *     ...
   *         size = put_system_header(ctx, buf_ptr, buf_end - buf_ptr, id);
   *
   * So the writer believed it had 128 bytes wherever it was called. The defect
   * is in what the callee was TOLD, not in what it wrote.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long total = 160;      /* the muxer's buffer */
  const long cursor = 96;      /* where the system header starts */
  const long remaining = total - cursor;   /* 64 */
  const long told = 128;       /* upstream's constant */
  CHECK(told > remaining, 781);            /* the claim, asserted */

  unsigned char *buf = calloc((size_t)total, 1);
  CHECK(buf, 782);

  const long cap_told = fixed ? remaining : told;
  long touched = -1;
  for (long i = 0; i < cap_told && cursor + i < total; i++) {
    touched = cursor + i;
    buf[cursor + i] = (unsigned char)0x5A;
  }

  /* ONE byte past, not the whole overrun. A write loop that really ran the full
   * extent corrupts the next chunk header and glibc aborts in free() -- rc 134,
   * which is an infrastructure failure, not a verdict. The crossing is what the
   * row measures; `extent` records how far the unreduced write would run. This
   * is the same shape case 3 uses. */
  if (cursor + cap_told > total) {
    touched = total;
    write_probe_u8(&buf[total], (unsigned char)0x5A);   /* the labelled crossing */
  }

  o->cap = (unsigned long)total;
  o->touched = touched;
  o->crossed = touched >= total;
  o->extent = told - remaining;
  o->damage = o->crossed;

  o->defect_text = "the system-header writer was initialised with a constant 128-byte capacity, so "
                   "at a cursor with less than that remaining it wrote past the muxer's buffer";
  o->fixed_text = "the fix passes buf_end - buf_ptr into the writer, so its capacity is the space "
                  "actually left";
  free(buf);
}
