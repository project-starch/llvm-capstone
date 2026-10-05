/* Case 0: b7946098b1 -- "swscale/alphablend: don't overread alpha plane on
 * subsampled odd size". On the last subsampled row ff_sws_alphablendaway averages
 * the alpha row BELOW the last one, which does not exist.
 *
 * Shape: a vertical average reads one row past the alpha plane, inside the
 *        frame's single AVBuffer
 * Consumer: libswscale/alphablend.c, ff_sws_alphablendaway()
 * SPATIAL and NESTED -- the plane is one of four carved from a single AVBuffer by
 * av_frame_get_buffer, so the crossing leaves the PLANE and not the allocation.
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

FFP_CASE(0) {
  o->defect_text = "the last row's vertical average read the alpha row below the "
                   "plane's last, inside the frame's one AVBuffer";
  o->fixed_text = "subsample_row is false on the last row, so only rows inside "
                  "the plane are averaged";

  /* YUVA420P: chroma is vertically subsampled (y_subsample = 1) while alpha is
   * not, which is the condition the defect needs. An odd height is the
   * "subsampled odd size" of the upstream subject. */
  const int w = 16, h = 5;
  AVFrame *f = av_frame_alloc();
  CHECK(f, 900);
  f->format = AV_PIX_FMT_YUVA420P;
  f->width = w;
  f->height = h;
  CHECK(av_frame_get_buffer(f, 0) >= 0, 901);

  /* NESTED, asserted rather than assumed: one buffer holds every plane. */
  CHECK(f->buf[0] && !f->buf[1], 902);
  const int alpha = 3;
  unsigned char *a = f->data[alpha];
  const ptrdiff_t alpha_step = f->linesize[alpha];
  unsigned char *plane_end = a + (ptrdiff_t)alpha_step * h;
  unsigned char *buf_base = f->buf[0]->data;
  unsigned char *buf_end = buf_base + f->buf[0]->size;
  CHECK(a >= buf_base && plane_end <= buf_end, 903);

  /* CONTAINED, asserted: a whole row past the plane is still inside the buffer,
   * which is what makes this a plane crossing and not a heap overflow. */
  o->plane_slack = (long)(buf_end - plane_end);
  o->contained = (plane_end + alpha_step) <= buf_end;
  CHECK(o->contained, 904);

  /* The alpha plane carries a known value; the slack after it carries another, so
   * a read that leaves the plane is distinguishable from one that does not. */
  for (int y = 0; y < h; y++)
    memset(a + (ptrdiff_t)alpha_step * y, 0x40, (size_t)w);
  memset(plane_end, 0xF0, (size_t)alpha_step);

  /* alphablend.c, the 8-bit path, reduced to the one average the fix changed:
   *
   *   y_subsample = 1 for YUVA420P's chroma loop, so ysrc runs over ceil(h/2)
   *   rows and the source row is (ysrc << y_subsample). On the LAST such row
   *   that index is h - 1 and the "+ alpha_step" row is h, one past the plane.
   *
   *   buggy  alpha = (a[2x] + a[2x+1] + 2 + a[2x+alpha_step] + a[2x+alpha_step+1]) >> 2
   *   fixed  subsample_row = y_subsample && (y << y_subsample) + 1 < lum_h  -> false
   *          here, so the two-tap horizontal average is used instead.
   */
  const int ysrc = (h - 1) >> 1;          /* the last subsampled row */
  const int row = ysrc << 1;              /* = 4 = h - 1, the plane's last row */
  CHECK(row == h - 1, 905);
  const int subsample_row = (row + 1) < h;  /* the fix's own condition */
  CHECK(!subsample_row, 906);               /* ... which is false exactly here */

  unsigned char *arow = a + (ptrdiff_t)alpha_step * row;
  const int x = 0;
  unsigned alpha_value;
  if (!fixed) {
    /* The pre-fix code takes the row below unconditionally when y_subsample. */
    o->crossed = 1;
    alpha_value = (read_probe(arow + 2 * x) + read_probe(arow + 2 * x + 1) + 2 +
                   read_probe(arow + 2 * x + alpha_step) +
                   read_probe(arow + 2 * x + alpha_step + 1)) >> 2;
  } else {
    o->crossed = 0;
    alpha_value = (read_probe(arow + 2 * x) + read_probe(arow + 2 * x + 1)) >> 1;
  }

  /* The consequence: the averaged alpha is pulled toward the slack's contents
   * instead of being the plane's own value. 0x40 in, 0xF0 outside. */
  o->damage = alpha_value != 0x40;

  av_frame_free(&f);
}
