#include "corpus.h"

/* libavcodec/aacpsy.c, psy_3gpp_init, fix e7a65142b972.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(10) {
  /* Case 10 -- psy_3gpp_init's error path, fix e7a65142b972. The freed object is
   * the psy private context; what is left holding its address is the FIELD
   * ctx->model_priv_data, because av_freep was pointed at the LOCAL.
   *
   * At the fix's parent:
   *
   *     av_freep(&pctx);
   *
   * and the fix points it at the owning field:
   *
   *     av_freep(&ctx->model_priv_data);
   *
   * av_freep clears whatever pointer it is handed; the defect is which one. */
  const unsigned long n = 48;
  unsigned char *alloc = malloc((size_t)n);
  CHECK(alloc, 901);
  memset(alloc, 0x11, (size_t)n);

  volatile unsigned char *model_priv_data = alloc;   /* the owning field */
  unsigned char *pctx = alloc;                       /* the local */
  free(pctx);
  pctx = NULL;                                       /* av_freep cleared the LOCAL */
  o->freed = 1;
  if (fixed)
    model_priv_data = NULL;                          /* av_freep(&ctx->model_priv_data) */

  unsigned char *fresh = malloc((size_t)n);          /* tcache hands back the same chunk */
  CHECK(fresh, 902);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = model_priv_data ? read_probe(model_priv_data) : 0u;   /* psy_3gpp_end's use */
  o->aliased = model_priv_data && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "av_freep cleared the local pctx while ctx->model_priv_data kept the released "
                   "address, so the teardown reaches storage that now belongs to another object";
  o->fixed_text = "the fix clears ctx->model_priv_data, the pointer that actually owns the object";
  free(fresh);
}
