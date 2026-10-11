#include "corpus.h"

/* libavformat/img2enc.c, write_packet's split-planes path, fix ca1c1f29ce47. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(16) {
  /* Case 16 -- img2enc's split-planes path, fix ca1c1f29ce47. PLAIN HEAP: the
   * three plane writes take their lengths from par->width/height, and read from
   * one packet buffer whose size is pkt->size.
   *
   * The fix adds the missing relation:
   *
   *     if (ysize + 2*usize + (desc->nb_components > 3) * ysize > pkt->size) {
   *         ret = AVERROR(EINVAL);
   *         goto fail;
   *     }
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long ysize = 32, usize = 8;      /* from the declared geometry */
  const unsigned long pkt_size = 32;              /* a size class, smaller than ysize + 2*usize */
  const unsigned long want = ysize + 2 * usize;   /* 48 */
  CHECK(want > pkt_size, 1061);                   /* the premise, asserted */

  unsigned char *data = calloc((size_t)pkt_size, 1);
  CHECK(data, 1062);

  long touched = 0;
  int crossed = 0;
  if (!fixed) {
    /* the three reads, at their successive offsets */
    const unsigned long offs[3] = {0, ysize, ysize + usize};
    const unsigned long lens[3] = {ysize, usize, usize};
    for (int i = 0; i < 3 && !crossed; i++) {
      for (unsigned long k = 0; k < lens[i]; k++) {
        const unsigned long at = offs[i] + k;
        if (at >= pkt_size) {
          touched = (long)at;
          crossed = 1;
          (void)read_probe_u8(data + at);         /* the labelled crossing */
          break;
        }
      }
    }
  }

  o->cap = pkt_size;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)want - (long)pkt_size;
  o->damage = crossed;

  o->defect_text = "the three plane reads took their lengths from the declared geometry, whose sum "
                   "exceeds the packet the data actually came in";
  o->fixed_text = "the fix rejects a packet smaller than the three plane sizes together";
  free(data);
}
