#include "corpus.h"
/* The plane control of the sublet-carve arm (FFP_SUBLET_CARVE): the same YUVA420P 16x5 frame as the
 * case, and one read one byte past the alpha plane's ALLOCATED extent (its padded_height rows) --
 * past what the port issued for the plane. It must fault at the probe: the port's bound exists. The
 * fixed arm reads the extent's last byte and must complete. */
FFP_CASE(99) {
  AVFrame *f = av_frame_alloc();
  CHECK(f, 990);
  f->format = AV_PIX_FMT_YUVA420P;
  f->width = 16;
  f->height = 5;
  CHECK(av_frame_get_buffer(f, 0) >= 0, 991);
  size_t extent = (size_t)f->linesize[3] * 32; /* FFALIGN(5, 32) rows */
  unsigned char *a = f->data[3];
  o->contained = 1;
  o->crossed = !fixed;
  (void)read_probe(a + (fixed ? extent - 1 : extent));
  o->defect_text = "control: one byte past the alpha plane's allocated extent";
  o->fixed_text = "control: the extent's last byte";
  av_frame_free(&f);
}
