#include "corpus.h"

/* libavcodec/diracdec.c, decode_lowdelay, fix 8a4ea9644833.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(8) {
  /* Case 8 -- decode_lowdelay's realloc, fix 8a4ea9644833. The freed object is
   * s->thread_buf's block: the realloc CONSUMED that field while publishing the
   * result as s->slice_params_buf, so thread_buf is left naming freed storage.
   *
   * At the fix's parent:
   *
   *     s->slice_params_buf = av_realloc_f(s->thread_buf, s->num_x * s->num_y, sizeof(DiracSlice));
   *
   * and the fix names the right source:
   *
   *     s->slice_params_buf = av_realloc_f(s->slice_params_buf, ...);
   */
  const unsigned long n = 48;
  /* slice_params_buf is allocated AFTER thread_buf, so thread_buf is not at the
   * top of the heap and realloc cannot grow it in place. That matters: if the
   * realloc extended the chunk instead of moving it, nothing would be freed and
   * the case would report INCONCLUSIVE for a reason unrelated to the defect. The
   * premise is asserted below rather than assumed. */
  unsigned char *thread_buf = malloc((size_t)n);
  CHECK(thread_buf, 881);
  memset(thread_buf, 0x11, (size_t)n);
  unsigned char *slice_params_buf = malloc((size_t)n);
  CHECK(slice_params_buf, 882);
  memset(slice_params_buf, 0x22, (size_t)n);
  /* A blocker after BOTH, so whichever field the arm reallocs, the block cannot
   * be grown in place and really is released. */
  unsigned char *blocker = malloc((size_t)n);
  CHECK(blocker, 887);

  /* The FIELD s->thread_buf keeps this address across the realloc. */
  volatile unsigned char *field_thread_buf = thread_buf;
  unsigned char *consumed_base = fixed ? slice_params_buf : thread_buf;

  /* The realloc: the buggy arm consumes thread_buf, the fixed arm its own field. */
  unsigned char *grown = realloc(consumed_base, (size_t)n * 64);
  CHECK(grown, 883);
  CHECK(grown != consumed_base, 884);   /* the premise: it moved, so the old block was freed */
  slice_params_buf = grown;
  o->freed = 1;
  if (fixed)
    field_thread_buf = thread_buf;      /* still live: the fix never consumed it */
  else
    thread_buf = NULL;                  /* its block is gone; the FIELD still names it */

  unsigned char *fresh = malloc((size_t)n);   /* reuses the released chunk */
  CHECK(fresh, 885);
  memset(fresh, 0xAA, (size_t)n);

  /* Any later use of s->thread_buf, which in the buggy arm names the chunk the
   * realloc released -- the chunk `fresh` now owns. In the fixed arm the field
   * still names thread_buf's own live block, whose bytes are 0x11. */
  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(field_thread_buf);   /* the labelled access */
  o->aliased = o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the realloc consumed s->thread_buf while publishing the result as "
                   "s->slice_params_buf, so thread_buf is left naming storage that now belongs to "
                   "another object";
  o->fixed_text = "the fix reallocs s->slice_params_buf, so each field owns its own block";
  free(fresh);
  free(slice_params_buf);
  free(blocker);
  if (thread_buf)
    free(thread_buf);
}
