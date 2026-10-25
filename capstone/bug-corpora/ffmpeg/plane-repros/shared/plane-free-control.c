#include "corpus.h"
/* The free control of the virtual carve arm (FFP_VIRTUAL_CARVE, shared/virtual-frame.c): a plane read
 * through the frame's own pointer AFTER the frame is freed. Freeing the frame frees its buffer, which
 * ends the frame's lifetime and every plane's, so the stale read must fault at the probe. The fixed
 * arm reads before the free. */
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
