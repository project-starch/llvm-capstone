#include "corpus.h"

/* prores_raw's frame header parse, fix 041d4f010e. The packet buffer is ONE
 * direct allocation; here it is the platform's calloc. */

FFH_CASE(6) {
  /* Case 6 -- prores_raw decode_frame, fix 041d4f010e. PLAIN HEAP: a header
   * length read FROM THE STREAM is consumed without checking how many bytes the
   * buffer actually holds.
   *
   * The guard at the fix's parent tested only the lower bound:
   *
   *     if (header_len < 62)
   *         return AVERROR_INVALIDDATA;
   *
   * and the fix adds the upper one:
   *
   *     if (header_len < 62 || bytestream2_get_bytes_left(&gb) < header_len - 2)
   *         return AVERROR_INVALIDDATA;
   *
   * So a stream declaring a header longer than the packet makes the parse walk
   * off the end. The crossing's LENGTH is chosen by the input, which is why the
   * fix bounds it rather than clamping the read.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long avail = 48;        /* bytes actually in the packet */
  const long header_len = 70;   /* what the stream declares; >= 62 passes the old guard */

  /* The claim, asserted rather than assumed: the old guard admits this header. */
  CHECK(header_len >= 62, 761);
  CHECK(header_len - 2 > avail, 762);   /* and the buffer cannot hold it */

  unsigned char *buf = calloc((size_t)avail, 1);
  CHECK(buf, 763);
  for (long i = 0; i < avail; i++)
    buf[i] = (unsigned char)(0x10 + i);

  /* The fix refuses the frame; the pre-fix code walks header_len - 2 bytes. */
  const long want = header_len - 2;
  long touched = -1;
  unsigned acc = 0;
  if (!fixed) {
    for (long i = 0; i < want; i++) {
      touched = i;
      if (i >= avail) {
        acc += read_probe_u8(&buf[i]);  /* the labelled crossing */
        if (i >= avail + 32)            /* the reduction's own stop */
          break;
      } else {
        acc += buf[i];
      }
    }
  } else {
    touched = -1;                       /* rejected before any read */
  }
  (void)acc;

  o->cap = (unsigned long)avail;
  o->touched = touched;
  o->crossed = touched >= avail;
  o->extent = want - avail;
  o->damage = o->crossed;

  o->defect_text = "the header-length guard checked only that header_len was at least 62, so a "
                   "stream declaring more than the packet holds walked the parse past the buffer";
  o->fixed_text = "the fix also requires bytestream2_get_bytes_left to cover header_len - 2, so an "
                  "over-long header is rejected before any read";
  free(buf);
}
