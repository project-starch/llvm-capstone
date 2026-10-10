#include "corpus.h"

/* vf_vif's filter mirror, fix 56309e476a. The scratch row is ONE direct
 * allocation per thread -- `s->temp[i] = av_calloc(s->width, sizeof(float))` at
 * libavfilter/vf_vif.c:517 -- and the hand-rolled mirror can hand back a
 * NEGATIVE index. Here it is the platform's calloc. */

FFH_CASE(2) {
  /* Case 2 -- vf_vif vif_filter1d, fix 56309e476a. PLAIN HEAP: a four-byte read
   * BELOW the base of a direct av_calloc.
   *
   * The pin mirrors an out-of-range tap index by hand (vf_vif.c:270):
   *
   *     jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);
   *     img_coeff = temp[jj];
   *
   * The reflection `2*w - jj - 1` is only correct while `jj < 2*w`. The tap
   * index is `jj = j - filt_w / 2 + filt_j`, so it reaches `w - 1 + filt_w / 2`,
   * and for a filter wide relative to the row -- the "small dimensions" of the
   * fix's subject line -- that exceeds `2*w`. At `jj == 2*w` the reflection
   * yields exactly -1, and the ternary has already taken its `jj >= w` branch,
   * so the negative value is never re-checked. The read is `temp[-1]`.
   *
   * The fix replaces the expression with `avpriv_mirror(jj, w - 1)`, a shared
   * helper that folds repeatedly instead of once.
   *
   * THIS ONE CROSSES BELOW THE BASE, and it is the only row in this corpus that
   * does. That matters for what can catch it: a bounds check that only tests the
   * upper limit passes it, while a capability's lower bound does not. ASan's
   * left redzone does catch it, which the runner measures. */
  const long w = 4;         /* a small row, as the fix's subject requires */
  const long filt_w = 17;   /* a wide filter: w - 1 + filt_w/2 == 11 >= 2*w */

  float *temp = calloc((size_t)w, sizeof *temp);
  CHECK(temp, 721);
  for (long i = 0; i < w; i++)
    temp[i] = 1.0f + (float)i;

  /* The tap that reaches exactly 2*w, which is the index whose reflection is
   * exactly -1. j is the output column, filt_j the tap: 0 - 17/2 + 16 == 8.
   * Taps beyond this one reflect further negative still, which is what `extent`
   * records; this is the first. */
  const long j = 0, filt_j = filt_w - 1;
  long jj_raw = j - filt_w / 2 + filt_j;
  CHECK(jj_raw == 2 * w, 722); /* the exact value that reflects to -1 */

  long jj;
  if (fixed) {
    /* avpriv_mirror(x, m): fold repeatedly into [0, m]. */
    long m = w - 1, x = jj_raw;
    while (x < 0 || x > m)
      x = x < 0 ? -x : 2 * m - x;
    jj = x;
  } else {
    jj = jj_raw < 0 ? -jj_raw : (jj_raw >= w ? 2 * w - jj_raw - 1 : jj_raw);
  }

  o->cap = (unsigned long)w;
  o->touched = jj;
  o->crossed = jj < 0 || jj >= w;
  /* Unreduced, every tap from 2*w upward reflects negative, so the span below
   * the base is filt_w/2 - w - 1 + 1 elements for this geometry. */
  o->extent = jj < 0 ? -jj : 0;

  float v = read_probe(&temp[jj]);
  o->damage = v != 0.0f && (jj < 0 || jj >= w);

  o->defect_text = "the hand-rolled mirror reflected a tap index of 2*w to -1 and never "
                   "re-checked it, so the read landed one element BELOW av_calloc's base";
  o->fixed_text = "the fix's avpriv_mirror folds repeatedly, so every index lands in [0, w-1]";
  free(temp);
}
