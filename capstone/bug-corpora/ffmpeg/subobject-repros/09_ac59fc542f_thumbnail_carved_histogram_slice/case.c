#include "corpus.h"
#include <stdint.h>

/* The ONE case in this corpus whose inner bound is a CARVED SLICE rather than a
 * declared struct member, and the reason it lives here rather than in a corpus
 * of its own: the boundary is still "inside one allocation".
 *
 * vf_thumbnail.c carves its per-thread and per-plane histograms out of a single
 * allocation by pointer arithmetic, two levels deep:
 *
 *     int *hist  = s->thread_histogram + HIST_SIZE * jobnr;   :206
 *     int *hhist = hist + 256 * plane;                        :259
 *     get_hist16(hhist, p, linesize, planewidth, h, s->bitdepth - 8);  :262
 *
 * HIST_SIZE is 3*256 (:38). So one av_calloc holds nb_threads slices of 768
 * ints; each slice holds three sub-slices of 256. NOTHING declares those
 * boundaries -- they are products of a multiplication -- which is precisely why
 * no allocator can know them. */
#define HIST_SIZE (3 * 256)
#define PLANE_HIST 256
#define NB_THREADS 4

__attribute__((noinline, used)) static void
bump_probe(volatile int *p) {
  *p = *p + 1;
}

/* get_hist16's tail loop, which is the whole defect (vf_thumbnail.c:190-192):
 *
 *     for (int x = width4; x < width; x++)
 *         hist[p16[x]]++;          <- pre-fix
 *         hist[(uint8_t)(p16[x] >> shift)]++;   <- post-fix
 *
 * p16[x] is a raw 16-bit sample. The histogram it indexes has 256 entries. The
 * main vectorised loop above it shifts and masks correctly; only the tail --
 * reached when the width is not a multiple of 4, i.e. "odd sized HBD inputs" --
 * forgets to, which is why the defect needs an odd width to show. */
static void hist_tail(int *hist, const uint16_t *p16, int width4, int width,
                      int shift, int fixed) {
  for (int x = width4; x < width; x++) {
    unsigned idx = fixed ? (unsigned)(uint8_t)(p16[x] >> shift) : p16[x];
    bump_probe(&hist[idx]);
  }
}

FF2_CASE(9) {
  /* Case 9 -- vf_thumbnail high-bit-depth histogram, fix ac59fc542f.
   * SUB-OBJECT, carved: a four-byte read-modify-WRITE from one carved sub-slice
   * into the next, inside a single allocation.
   *
   * A 10-bit sample has a value up to 1023 and is used UNSHIFTED as an index
   * into a 256-entry histogram. Index 1023 is 767 entries past the sub-slice --
   * which, since the sub-slices are contiguous, is 511 entries into the next
   * thread's slice. The fix shifts the sample down by bitdepth-8 and masks it to
   * a byte, so the index is 0..255 and lands in its own sub-slice.
   *
   * THE MAGNITUDE IS DATA-CONTROLLED, and is reported rather than implied: a
   * 16-bit sample reaches index 65535, which is 65279 entries -- 261 116 bytes --
   * past the sub-slice. This case uses a 10-bit sample, which keeps the crossing
   * INSIDE the allocation so that what is measured is the carved-slice crossing
   * and not a crossing of the malloc bound. The larger magnitude is what makes
   * the same defect reachable past the allocation too; that is a different row
   * and is not claimed here.
   *
   * NOTHING WE HAVE CAN CATCH THIS: both the sub-slice and the slice boundaries
   * are arithmetic on one allocation's base, so a per-allocation bound is in
   * bounds for every one of these writes. */
  const size_t total_ints = (size_t)NB_THREADS * HIST_SIZE;
  int *thread_histogram = av_refstruct_allocz(total_ints * sizeof(int));
  CHECK(thread_histogram, 671);

  const int jobnr = 0, plane = 0;
  int *hist = thread_histogram + HIST_SIZE * jobnr;
  int *hhist = hist + PLANE_HIST * plane;
  /* The claim, asserted: the sub-slice's end IS the next sub-slice's start, and
   * both are interior to one allocation. Written against plane 1 so the
   * assertion is not vacuously true at plane 0. */
  CHECK(hhist + PLANE_HIST == hist + PLANE_HIST * (plane + 1), 672);
  CHECK(hhist >= thread_histogram
            && hhist + PLANE_HIST <= thread_histogram + total_ints,
        673);

  /* A 10-bit sample, and a width whose tail is not a multiple of 4 -- the "odd
   * sized HBD input" of the fix's subject line. */
  const int bitdepth = 10, shift = bitdepth - 8;
  const uint16_t sample = 1023; /* the maximum a 10-bit sample can hold */
  const int width = 5, width4 = 4;
  uint16_t p16[5] = {0, 0, 0, 0, sample};

  hist_tail(hhist, p16, width4, width, shift, fixed);

  /* Where the bump landed. Under the fix it must be inside the sub-slice; at the
   * pin it must be past it and still inside the allocation. */
  const unsigned idx_buggy = sample;
  const unsigned idx_fixed = (uint8_t)(sample >> shift);
  int bumped_outside = hhist[idx_buggy] == 1 && idx_buggy >= PLANE_HIST;
  int bumped_inside = hhist[idx_fixed] == 1 && idx_fixed < PLANE_HIST;
  long past_entries = (long)idx_buggy - PLANE_HIST + 1;
  int still_in_allocation =
      (hhist + idx_buggy) < (thread_histogram + total_ints);

  printf("cap_entries=%d sub_slice=%d index=%u past_entries=%ld "
         "unreduced_past_bytes=%ld in_allocation=%d\n",
         HIST_SIZE, PLANE_HIST, idx_buggy, past_entries,
         (long)(65535 - PLANE_HIST + 1) * (long)sizeof(int), still_in_allocation);

  FF2_VERDICT(!fixed && bumped_outside && still_in_allocation,
              fixed && bumped_inside,
              "a 10-bit sample was used unshifted as an index, so the bump landed 768 "
              "entries past its 256-entry carved sub-slice, inside one av_calloc",
              "the fix shifts by bitdepth-8 and masks to a byte, keeping the index in its sub-slice");
  av_refstruct_unref(&thread_histogram);
  return !fixed ? !bumped_outside : !bumped_inside;
}
