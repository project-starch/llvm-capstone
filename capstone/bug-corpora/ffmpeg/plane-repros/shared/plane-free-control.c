#include "corpus.h"
/* The free control of the sublet-carve arm (FFP_SUBLET_CARVE): a plane read through the frame's own
 * pointer AFTER the frame is freed. Freeing the frame is the Sublet heap's revoke of the block, so the
 * stale read must fault at the probe (a revoked alias). The fixed arm reads before the free. */
FFP_CASE(98) {
  AVFrame *f = av_frame_alloc();
  CHECK(f, 980);
  f->format = AV_PIX_FMT_YUVA420P;
  f->width = 16;
  f->height = 5;
  CHECK(av_frame_get_buffer(f, 0) >= 0, 981);
  unsigned char *volatile stale = f->data[3];
  o->contained = 1;
  o->crossed = !fixed;
  if (fixed)
    (void)read_probe(stale);
  av_frame_free(&f);
  if (!fixed)
    (void)read_probe(stale);
  o->defect_text = "control: an alpha-plane read after av_frame_free";
  o->fixed_text = "control: the same read before the free";
}
