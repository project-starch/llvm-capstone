#include "corpus.h"

/* avf_showcwt's kernel scan, fix d133b4a231. The buffer is ONE direct
 * allocation -- `tkernel = av_malloc_array(size, sizeof(*tkernel))` at
 * avf_showcwt.c:725 -- and the backward scan starts one element past its end.
 *
 * Here it is the platform's calloc, for the reason corpus.h states. */

FFH_CASE(0) {
  /* Case 0 -- avf_showcwt compute_kernel, fix d133b4a231. PLAIN HEAP: a
   * four-byte read one element past a direct av_malloc_array.
   *
   * The forward scan is exclusive and the backward scan is not
   * (avf_showcwt.c:747-762 at the pin):
   *
   *     for (int n = a; n < b; n++)            // writes, EXCLUSIVE of b
   *         tkernel[n+range] = ff;
   *     ...
   *     for (int n = b; n >= a; n--)           // reads, INCLUSIVE of b
   *         if (tkernel[n+range] != 0.f) { stop = n; break; }
   *
   * with `b = FFMIN(frequency + 12*sqrtf(1/deviation) - 0.5f, size + a)` and
   * `range = -a` (:736-738). b is clamped to `size + a`, so `b + range` is at
   * most `size + a - a` = size -- exactly one element past a `size`-element
   * array. The fix starts the backward scan at `b - 1`.
   *
   * The upstream commit says it was "Reproduced with a small output (e.g.
   * size=2x2) under ASan", and small is what makes b hit its clamp.
   *
   * THIS ONE LEAVES THE ALLOCATION, which is what separates this corpus from
   * its sub-object siblings: a per-object bound is NOT in bounds for it, so
   * ASan sees it and so should a capability machine. */
  const long size = 4; /* the clamped, small case the fix's report names */
  const long a = -2;
  const long range = -a;
  const long b = size + a; /* the clamp at :737, i.e. b + range == size */

  float *tkernel = calloc((size_t)size, sizeof *tkernel);
  CHECK(tkernel, 701);
  /* The claim, asserted: the scan's first index IS one past the array. */
  CHECK(b + range == size, 702);

  /* The forward pass, exclusive of b, exactly as upstream writes it. */
  for (long n = a; n < b; n++)
    tkernel[n + range] = 1.0f;

  long first = fixed ? (b - 1) : b; /* the fix moves the start down by one */
  o->cap = (unsigned long)size;
  o->touched = first + range;
  o->crossed = o->touched >= size;
  /* How far the unreduced scan would run: it breaks on the first non-zero, and
   * index size-1 was just written, so exactly one element is out of bounds. */
  o->extent = 1;

  /* The read itself, on the labelled probe so a report can be required here. */
  float v = read_probe(&tkernel[first + range]);
  o->damage = o->crossed && v != 0.0f; /* upstream's `stop` would take the OOB n */

  o->defect_text = "the backward kernel scan started at b, so it read tkernel[size] -- "
                   "one element past av_malloc_array(size, 4)";
  o->fixed_text = "the fix starts the scan at b - 1, so the first read is the array's last element";
  free(tkernel);
}
