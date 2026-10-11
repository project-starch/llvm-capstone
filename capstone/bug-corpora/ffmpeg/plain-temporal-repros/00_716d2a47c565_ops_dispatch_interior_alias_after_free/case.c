#include "corpus.h"

/* libswscale/ops_dispatch.c, compile_single, fix 716d2a47c565.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(0) {
  /* Case 0 -- compile_single's interior alias, fix 716d2a47c565. The freed
   * object is the pass struct `p`; the pointer left holding its address is
   * `comp`, an INTERIOR pointer at &p->comp. The later access reads
   * `comp->backend->flags` through it.
   *
   * At the fix's parent:
   *
   *     if (ret >= 0) {
   *         (*output)->backend = comp->backend->flags;
   *
   * and the fix reads the same field from the local copy instead:
   *
   *         (*output)->backend = c.backend->flags;
   *
   * so the fixed arm never follows a pointer into the freed block. */
  const unsigned long n = 48;
  const unsigned long comp_off = 16;              /* where &p->comp sits inside p */
  unsigned char *p = malloc((size_t)n);
  CHECK(p, 801);
  memset(p, 0x11, (size_t)n);

  volatile unsigned char *comp = p + comp_off;    /* the interior alias */
  free(p);
  o->freed = 1;
  if (fixed)
    comp = NULL;                                  /* the fix reads `c`, not the freed block */

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 802);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = comp ? read_probe(comp) : 0u;     /* the labelled access */
  o->aliased = comp && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "comp is an interior pointer into the freed pass struct, so reading "
                   "comp->backend->flags reads storage that now belongs to another object";
  o->fixed_text = "the fix reads the field from the local copy c, which is not part of the freed "
                  "allocation";
  free(fresh);
}
