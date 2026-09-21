/* Case 2: 316531e61c -- vidstabtransform parks a raw plane pointer
 *
 * Shape: parked pointer / reuse / stale write
 * Consumer: libavfilter/vf_vidstabtransform.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

/* libvidstab's transform state, reduced to the field that matters: a shallow
 * copy of the source frame, held with no reference taken. */
struct vs_state {
  unsigned char *src;
};

FF2_CASE(2) {
  struct vs_state td = {0};
  AVBufferRef *in = av_buffer_pool_get(g_pool);
  CHECK(in, 630);
  memset(in->data, 0x11, POOL_BYTES);
  td.src = in->data;    /* vsTransformPrepare's separate-buffer path */
  av_buffer_unref(&in); /* the filter returns the source frame */
  if (fixed)
    td.src = NULL; /* always the in-place path: nothing is carried over */

  AVBufferRef *next = av_buffer_pool_get(g_pool);
  CHECK(next, 631);
  memset(next->data, 0x22, POOL_BYTES);

  /* the next frame: vsFrameIsNull(&td.src) is false, so the allocation is
   * skipped and vsFrameCopy writes through whatever td.src still names */
  int wrote_through_stale = 0;
  if (td.src) {
    td.src[0] = 0x99;
    wrote_through_stale = 1;
  }
  printf("wrote_through_stale=%d new_owner_byte=0x%02X\n", wrote_through_stale,
         next->data[0]);
  FF2_VERDICT(!fixed && wrote_through_stale && next->data[0] == 0x99,
              fixed && next->data[0] == 0x22,
              "the parked pointer corrupts the new owner",
              "nothing is carried across frames");
  int bad = !fixed ? !(wrote_through_stale && next->data[0] == 0x99)
                   : next->data[0] != 0x22;
  av_buffer_unref(&next);
  return bad;
}
