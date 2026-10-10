#include "corpus.h"

/* libavfilter/afir_template.c, fir_quantums' crossfade, fix 3fbb9560821b. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(21) {
  /* Case 21 -- afir's crossfade, fix 3fbb9560821b. PLAIN HEAP: the copy takes a
   * fixed min_part_size samples from the input frame at `offset`, bounded by
   * nothing against the frame's own nb_samples.
   *
   * At the fix's parent:
   *
   *     memcpy(dst, in, sizeof(ftype) * min_part_size);
   *
   * and the fix clamps it:
   *
   *     const int nb_samples = FFMIN(min_part_size, s->in->nb_samples - offset);
   *     ...
   *     memcpy(dst, in, sizeof(ftype) * nb_samples);
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long nb_samples = 4;             /* the frame's own sample count */
  const unsigned long offset = 2;                 /* the final partition's start */
  const unsigned long min_part_size = 4;          /* the configured partition length */
  const unsigned long elems = nb_samples;         /* 4 floats == 16 B: a size class */
  CHECK(offset + min_part_size > nb_samples, 1111);   /* the premise, asserted */

  float *in = calloc((size_t)elems, sizeof *in);
  CHECK(in, 1112);
  for (unsigned long i = 0; i < elems; i++)
    in[i] = (float)i;

  const unsigned long n = fixed ? (nb_samples - offset) : min_part_size;
  long touched = 0;
  int crossed = 0;
  for (unsigned long i = 0; i < n; i++) {
    const unsigned long at = offset + i;
    if (at >= elems) {
      touched = (long)at;
      crossed = 1;
      (void)read_probe(in + at);                  /* the labelled crossing */
      break;
    }
  }

  o->cap = elems;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)(offset + min_part_size) - (long)nb_samples;
  o->damage = crossed;

  o->defect_text = "the crossfade copied a fixed min_part_size samples from the frame at offset, "
                   "with nothing bounding it against the frame's own nb_samples";
  o->fixed_text = "the fix clamps the length to FFMIN(min_part_size, nb_samples - offset)";
  free(in);
}
