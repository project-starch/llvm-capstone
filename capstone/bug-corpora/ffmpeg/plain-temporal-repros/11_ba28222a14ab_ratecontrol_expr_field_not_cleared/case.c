#include "corpus.h"

/* libavcodec/ratecontrol.c, ff_rate_control_uninit, fix ba28222a14ab.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(11) {
  /* Case 11 -- ff_rate_control_uninit's expression field, fix ba28222a14ab. The
   * freed object is the rate-control expression; what is left holding its
   * address is rcc->rc_eq_eval, because av_expr_free does not clear its
   * argument.
   *
   * The fix makes the teardown idempotent:
   *
   *     av_expr_free(rcc->rc_eq_eval);
   *     rcc->rc_eq_eval = NULL;
   *
   * The later access is this SAME function run a second time: the mpeg encoders
   * set FF_CODEC_CAP_INIT_CLEANUP, so an init failure runs uninit and
   * ff_mpv_encode_end runs it again. */
  const unsigned long n = 48;
  unsigned char *expr = malloc((size_t)n);
  CHECK(expr, 911);
  memset(expr, 0x11, (size_t)n);

  volatile unsigned char *rc_eq_eval = expr;      /* the field */
  free(expr);                                     /* av_expr_free: releases, does not clear */
  o->freed = 1;
  if (fixed)
    rc_eq_eval = NULL;                            /* the fix's rcc->rc_eq_eval = NULL */

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 912);
  memset(fresh, 0xAA, (size_t)n);

  /* The second entry to uninit, reached through FF_CODEC_CAP_INIT_CLEANUP. */
  o->bytes = n;
  o->marker = 0xAA;
  o->observed = rc_eq_eval ? read_probe(rc_eq_eval) : 0u;   /* the labelled access */
  o->aliased = rc_eq_eval && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "av_expr_free does not clear its argument, so a second run of the same uninit "
                   "reaches storage that now belongs to another object";
  o->fixed_text = "the fix nulls rcc->rc_eq_eval, making the teardown idempotent";
  free(fresh);
}
