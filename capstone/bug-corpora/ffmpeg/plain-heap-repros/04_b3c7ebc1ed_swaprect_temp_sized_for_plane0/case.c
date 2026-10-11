#include "corpus.h"

/* vf_swaprect's scratch row buffer, fix b3c7ebc1ed. The buffer is ONE direct
 * allocation: libavfilter/vf_swaprect.c:213 at the fix's parent is
 * `s->temp = av_malloc_array(inlink->w, s->pixsteps[0]);`. Here it is the
 * platform's calloc, because what the row turns on is the SIZE, not av_malloc. */

FFH_CASE(4) {
  /* Case 4 -- vf_swaprect config_input, fix b3c7ebc1ed. PLAIN HEAP: a row copy
   * for a CHROMA plane overruns a scratch buffer sized for the LUMA plane.
   *
   * The allocation at the fix's parent sizes the scratch for plane 0 alone:
   *
   *     s->temp = av_malloc_array(inlink->w, s->pixsteps[0]);
   *
   * but filter_frame copies a row of EVERY plane through it
   * (vf_swaprect.c:187-189 at the pin):
   *
   *     memcpy(s->temp, src, pw[p] * s->pixsteps[p]);
   *
   * For a semi-planar 4:2:0 format the two differ. NV12 has pixsteps {1, 2} and
   * log2_chroma_w 1, so for width w:
   *
   *     plane 0 needs  w * 1
   *     plane 1 needs  AV_CEIL_RSHIFT(w, 1) * 2  ==  2 * ceil(w / 2)
   *
   * Those are equal for even w and differ by ONE for odd w -- which is why
   * upstream's reproducer is named `odd17_nv12`. At w = 17 the buffer is 17
   * bytes and the chroma row is 18.
   *
   * The fix takes the max over every plane instead:
   *
   *     for (int p = 0; p < s->nb_planes; p++) { ... size = FFMAX(size, width * s->pixsteps[p]); }
   *     s->temp = av_malloc(size);
   *
   * THIS ONE LEAVES THE ALLOCATION: the write is past a direct allocation, not
   * between two members of one. */
  const long w = 17;                  /* upstream's odd width */
  const long pixstep_luma = 1;        /* NV12 plane 0 */
  const long pixstep_chroma = 2;      /* NV12 plane 1, interleaved U/V */
  const long chroma_w = (w + 1) / 2;  /* AV_CEIL_RSHIFT(w, 1) */

  const long luma_row = w * pixstep_luma;              /* 17 */
  const long chroma_row = chroma_w * pixstep_chroma;   /* 18 */
  /* The claim, asserted rather than assumed: the chroma row really is longer. */
  CHECK(chroma_row == luma_row + 1, 741);

  const long cap = fixed ? (luma_row > chroma_row ? luma_row : chroma_row) : luma_row;
  unsigned char *temp = calloc((size_t)cap, 1);
  CHECK(temp, 742);

  unsigned char *src = calloc((size_t)chroma_row, 1);
  CHECK(src, 743);
  for (long i = 0; i < chroma_row; i++)
    src[i] = (unsigned char)(0xA0 + i);

  /* The copy filter_frame makes for plane 1. Written a byte at a time so the
   * crossing lands on the labelled probe instead of inside libc's memcpy. */
  for (long i = 0; i < chroma_row; i++) {
    if (i >= cap)
      write_probe_u8(&temp[i], src[i]);   /* the labelled crossing */
    else
      temp[i] = src[i];
  }

  o->cap = (unsigned long)cap;
  o->touched = chroma_row - 1;
  o->crossed = o->touched >= cap;
  o->extent = chroma_row - luma_row;      /* one byte, for this width */
  o->damage = o->crossed;                 /* a write past the allocation */

  o->defect_text = "the scratch row was sized w * pixsteps[0], the luma plane, while the "
                   "chroma row of a semi-planar 4:2:0 format at odd width needs one byte "
                   "more -- so the copy wrote past av_malloc_array";
  o->fixed_text = "the fix sizes the scratch by the widest plane, so every row copy fits";
  free(src);
  free(temp);
}
