#include "../shared/corpus.h"

/* Case 12 -- vf_thumbnail's high-bit-depth histogram, fix ac59fc542f
 * ("avfilter/vf_thumbnail: fix buffer overflow for odd sized HBD inputs").
 * Moved here on 2026-10-10 from ../../subobject-repros/09 (that corpus's
 * readings of it stay in its results/), because the region it leaves is a CARVE,
 * not a struct member, and this corpus is where a carve is the inner allocator.
 *
 * ONE allocation carved TWO levels deep (libavfilter/vf_thumbnail.c at the
 * fix's parent):
 *
 *     s->thread_histogram = av_calloc(HIST_SIZE, s->nb_threads * sizeof(*s->thread_histogram)); :337
 *     int *hist  = s->thread_histogram + HIST_SIZE * jobnr;                                       :206
 *     int *hhist = hist + 256 * plane;                                                             :259
 *
 * HIST_SIZE is 3*256 (:38): one slice of 768 ints per thread, three 256-entry
 * plane sub-slices in each. get_hist16's tail loop indexes the sub-slice with
 * an UNSHIFTED sample (:190-192):
 *
 *     for (int x = width4; x < width; x++)
 *         hist[p16[x]]++;                         <- at the parent
 *         hist[(uint8_t)(p16[x] >> shift)]++;     <- the fix
 *
 * reached only when the width is not a multiple of 4. A 10-bit sample of 1023
 * lands 767 entries past its 256-entry sub-slice: 511 entries into the NEXT
 * thread's slice, still inside the one allocation. (A 16-bit sample reaches
 * 65535 and leaves the allocation; that is a different row, not claimed here.)
 *
 * The carves: every plane sub-slice of every thread is one ffc_carve(), in
 * ascending order, 1024 bytes each, so a thread's slice is three consecutive
 * carves -- both levels of upstream's arithmetic, expressed at the finer one,
 * which is the bound the defect crosses. */
#define HIST_SIZE (3 * 256)
#define PLANE_HIST 256
#define NB_THREADS 4

FFC_CASE(12) {
  const size_t total = (size_t)NB_THREADS * HIST_SIZE;
  const size_t bytes = total * sizeof(int);
  int *thread_histogram = calloc(HIST_SIZE, NB_THREADS * sizeof(int)); /* :337 */
  CHECK(thread_histogram, 1201);

  static char names[NB_THREADS * 3][32];
  int *sub[NB_THREADS * 3];
  for (int j = 0; j < NB_THREADS; j++)
    for (int p = 0; p < 3; p++) {
      int k = j * 3 + p;
      snprintf(names[k], sizeof names[k], "hist[t%d][plane%d]", j, p);
      sub[k] = ffc_carve(thread_histogram, ((size_t)j * HIST_SIZE + (size_t)p * PLANE_HIST) * sizeof(int),
                         PLANE_HIST * sizeof(int), names[k]);
    }

  const int jobnr = 0, plane = 0;
  int *hhist = sub[jobnr * 3 + plane];

  /* A 10-bit sample, and a width whose tail is not a multiple of 4. */
  const int bitdepth = 10, shift = bitdepth - 8;
  const unsigned sample = 1023;               /* the maximum a 10-bit sample holds */
  const unsigned idx = fixed ? (unsigned)(uint8_t)(sample >> shift) : sample;

  /* The bump: hist[idx]++, a read-modify-write of one int. The first byte it
   * touches is the read. On the buggy arm that is 767 entries past the
   * sub-slice, which the whole access never takes out of the allocation. */
  volatile uint32_t *at = (volatile uint32_t *)&hhist[idx];
  ffc_note(o, thread_histogram, bytes, hhist, PLANE_HIST * sizeof(int), (const void *)at, sizeof(int));
  uint32_t v = read_probe_u32(at);
  write_probe_u32(at, v + 1);

  /* The consequence the report describes: a count lands in another thread's
   * histogram (thread 1, plane 0, entry 255), which merges into the frame's. */
  o->damage = !fixed && sub[1 * 3 + 0][PLANE_HIST - 1] == 1;  /* through thread 1's own carve:
                                                              under the Sublet carve the block
                                                              pointer is a token, never read */

  o->defect_text = "a 10-bit sample indexed the 256-entry plane histogram unshifted, so the "
                   "bump landed 767 entries past its carved sub-slice, in thread 1's slice";
  o->fixed_text = "the fix shifts by bitdepth-8 and masks to a byte, keeping the index in its sub-slice";
  free(thread_histogram);
}
