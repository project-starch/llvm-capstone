#include "corpus.h"

/* libavutil/tx.c, ff_tx_init_subtx, fix 265731f201f1.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(9) {
  /* Case 9 -- ff_tx_init_subtx's failure path, fix 265731f201f1. The freed
   * object is the subcontext; what is left holding its address is the FIELD
   * s->sub, because the free went through a LOCAL copy.
   *
   * At the fix's parent:
   *
   *     av_free(sub);
   *
   * and the fix frees through the field, which also clears it:
   *
   *     av_freep(&s->sub);
   *
   * This is exactly the distinction this corpus's gate turns on. */
  const unsigned long n = 48;
  unsigned char *sub_alloc = malloc((size_t)n);
  CHECK(sub_alloc, 891);
  memset(sub_alloc, 0x11, (size_t)n);

  volatile unsigned char *s_sub = sub_alloc;      /* the owning field */
  unsigned char *sub = sub_alloc;                 /* the local copy the free used */
  free(sub);                                      /* av_free(sub) */
  o->freed = 1;
  if (fixed)
    s_sub = NULL;                                 /* av_freep(&s->sub) clears the field too */

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 892);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = s_sub ? read_probe(s_sub) : 0u;   /* teardown's use of s->sub */
  o->aliased = s_sub && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the failure path freed the subcontext through a local copy, so s->sub kept the "
                   "released address and teardown reaches storage that now belongs to another "
                   "object";
  o->fixed_text = "the fix frees through av_freep(&s->sub), which clears the owning field";
  free(fresh);
}
