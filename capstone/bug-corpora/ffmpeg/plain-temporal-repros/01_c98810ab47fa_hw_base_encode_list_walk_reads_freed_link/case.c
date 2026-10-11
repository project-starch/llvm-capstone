#include "corpus.h"

/* libavcodec/hw_base_encode.c, ff_hw_base_encode_close, fix c98810ab47fa.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(1) {
  /* Case 1 -- the close path's list walk, fix c98810ab47fa. The freed object is
   * a list node; the pointer left holding its address is the loop variable
   * itself, and the later access is the INCREMENT, `pic = pic->next`.
   *
   * At the fix's parent:
   *
   *     for (pic = ctx->pic_start; pic; pic = pic->next)
   *         base_encode_pic_free(pic);
   *
   * and the fix latches the link before the free:
   *
   *     for (... *pic = ctx->pic_start, *next_pic = pic; pic; pic = next_pic) {
   *         next_pic = pic->next;
   *         base_encode_pic_free(pic);
   *     }
   */
  const unsigned long n = 48;
  unsigned char *node = malloc((size_t)n);        /* the node the loop frees */
  CHECK(node, 811);
  memset(node, 0x11, (size_t)n);

  unsigned long latched_ok = 0;
  if (fixed) {
    (void)node[0];                                /* the fix reads the link BEFORE the free */
    latched_ok = 1;
  }
  free(node);
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 812);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  /* The increment's read of `pic->next`, at the node's first word. */
  o->observed = latched_ok ? 0u : read_probe(node);   /* the labelled access */
  o->aliased = !latched_ok && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the loop's increment read pic->next out of the node it had just freed, so the "
                   "walk continues through storage that now belongs to another object";
  o->fixed_text = "the fix latches next_pic before calling the free, so no link is read from freed "
                  "storage";
  free(fresh);
}
