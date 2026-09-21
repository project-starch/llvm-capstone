/* Case 1: 1886c3269d -- h264_refs clears ref_list only up to ref_count
 *
 * Shape: partial clear / reuse / stale read
 * Consumer: libavcodec/h264_refs.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

#define SLOTS 8

/* H264Ref carries data[3] into a picture's planes and a parent pointer. */
struct h264_ref {
  unsigned char *data;
  int valid;
};

FF2_CASE(1) {
  struct h264_ref ref_list[SLOTS] = {0};
  AVBufferRef *pic[SLOTS];
  const int ref_count = 6, len = 2; /* the list shrank: two entries stay valid */

  for (int i = 0; i < SLOTS; i++) {
    CHECK((pic[i] = av_buffer_pool_get(g_pool)), 620);
    pic[i]->data[0] = (unsigned char)(0x40 + i);
    ref_list[i].data = pic[i]->data;
    ref_list[i].valid = 1;
  }
  /* the reset, with upstream's bound and with the fix */
  for (int i = len; i < (fixed ? SLOTS : ref_count); i++)
    memset(&ref_list[i], 0, sizeof(ref_list[i]));
  /* every picture past the live length is returned to the pool */
  for (int i = len; i < SLOTS; i++)
    av_buffer_unref(&pic[i]);

  int survivors = 0;
  for (int i = len; i < SLOTS; i++)
    survivors += ref_list[i].valid;

  AVBufferRef *reused = av_buffer_pool_get(g_pool);
  CHECK(reused, 621);
  memset(reused->data, 0xCC, POOL_BYTES);
  /* slot 7 is past ref_count, so upstream's bound never cleared it */
  unsigned char *stale = ref_list[SLOTS - 1].data;
  int same = stale == reused->data;
  CHECK(ref_list[0].data && ref_list[0].data[0] == 0x40, 623); /* a live record */

  printf("survivors_past_len=%d reuse_same_address=%d stale_read=0x%02X\n",
         survivors, same, stale ? stale[0] : 0);
  FF2_VERDICT(!fixed && survivors && same && stale[0] == 0xCC, fixed && !survivors,
              "a record past ref_count still names reissued storage",
              "the reset covers the whole list");
  int bad = !fixed ? !(survivors && same && stale[0] == 0xCC) : !!survivors;
  av_buffer_unref(&reused);
  for (int i = 0; i < len; i++)
    av_buffer_unref(&pic[i]);
  return bad;
}
