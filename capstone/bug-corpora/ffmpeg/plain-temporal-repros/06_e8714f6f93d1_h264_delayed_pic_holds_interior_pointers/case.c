#include "corpus.h"

/* libavcodec/h264.c, ff_h264_free_tables, fix e8714f6f93d1.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(6) {
  /* Case 6 -- ff_h264_free_tables and the reorder array, fix e8714f6f93d1. The
   * freed object is the DPB, ONE av_mallocz_array block; what is left holding
   * its address is delayed_pic[], an array of INTERIOR pointers &DPB[i].
   *
   * The fix clears that array before the free:
   *
   *     memset(h->delayed_pic, 0, sizeof(h->delayed_pic));
   *     av_freep(&h->DPB);
   *
   * THIS ONE WRITES through the stale interior pointer. */
  const unsigned long n = 48;
  const unsigned long pic_off = 16;               /* where &DPB[i] falls inside the block */
  unsigned char *dpb = malloc((size_t)n);
  CHECK(dpb, 861);
  memset(dpb, 0x11, (size_t)n);
  volatile unsigned char *delayed_pic = dpb + pic_off;   /* the interior pointer */

  if (fixed)
    delayed_pic = NULL;                           /* the fix's memset of delayed_pic */
  free(dpb);                                      /* av_freep(&h->DPB): the whole block */
  o->freed = 1;

  unsigned char *fresh = malloc((size_t)n);       /* tcache hands back the same chunk */
  CHECK(fresh, 862);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  if (delayed_pic)
    write_probe(delayed_pic, (unsigned char)0x5A);        /* `->reference = 0` */
  o->observed = fresh[pic_off];                   /* did the store land in the NEW object? */
  o->aliased = delayed_pic && o->observed == 0x5A;
  o->damage = o->aliased;
  o->defect_text = "delayed_pic held interior pointers into the DPB block that av_freep released, "
                   "so setting ->reference writes into storage that now belongs to another object";
  o->fixed_text = "the fix zeroes delayed_pic before freeing the DPB, so no interior pointer "
                  "survives";
  free(fresh);
}
