#include "corpus.h"

/* libavformat/rtpdec_av1.c, the OBU reassembly loop, fix 18761f9fb55c. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(17) {
  /* Case 17 -- rtpdec_av1's OBU skip, fix 18761f9fb55c. PLAIN HEAP: skipping a
   * temporal-delimiter or tile-list OBU advanced the OUTPUT cursor instead of
   * the INPUT one.
   *
   * At the fix's parent:
   *
   *     if ((obu_type == AV1_OBU_TEMPORAL_DELIMITER) ||
   *         (obu_type == AV1_OBU_TILE_LIST)) {
   *         pktpos += obu_size;
   *         rem_pkt_size -= obu_size;
   *
   * and the fix advances the input cursor:
   *
   *         buf_ptr += obu_size;
   *
   * No byte is written for the skipped OBU, so the destination offset outruns
   * the packet by exactly its size.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long pkt_size = 16;              /* a size class */
  const unsigned long obu_size = 16;              /* the skipped OBU */
  unsigned char *data = calloc((size_t)pkt_size, 1);
  CHECK(data, 1071);

  unsigned long pktpos = 0, buf_ptr = 0;
  if (fixed)
    buf_ptr += obu_size;                          /* the fix: skip the INPUT */
  else
    pktpos += obu_size;                           /* the defect: skip the OUTPUT */
  (void)buf_ptr;

  /* The next OBU's first written byte lands at pktpos. */
  o->cap = pkt_size;
  o->touched = (long)pktpos;
  o->crossed = pktpos >= pkt_size;
  o->extent = (long)obu_size;
  o->damage = o->crossed;
  if (o->crossed)
    write_probe_u8(data + pktpos, 0xA5);          /* the labelled crossing */

  o->defect_text = "skipping an OBU advanced the output cursor instead of the input one, so the "
                   "next write lands past the packet by the skipped OBU's size";
  o->fixed_text = "the fix advances buf_ptr, the input cursor, leaving the output offset where it "
                  "belongs";
  free(data);
}
