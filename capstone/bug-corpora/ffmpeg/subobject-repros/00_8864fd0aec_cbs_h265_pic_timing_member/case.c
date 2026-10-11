#include "corpus.h"
#include <stdint.h>

/* The two adjacent members, exactly as cbs_h265.h:627-628 declares them at
 * n9.0.1. HEVC_MAX_SLICE_SEGMENTS is 600 (hevc.h:150). The whole struct is ONE
 * allocation, so the bound between these two members is not a bound any
 * allocator knows about. */
#define SEGMENTS 600
struct pic_timing {
  uint16_t num_nalus_in_du_minus1[SEGMENTS];
  uint32_t du_cpb_removal_delay_increment_minus1[SEGMENTS];
};

FF2_CASE(0) {
/* Case 0 -- CBS H.265 pic_timing SEI, fix 8864fd0aec. SUB-OBJECT: a two-byte
 * write from one struct member into the next, inside a single refstruct
 * allocation.
 *
 * The syntax reader bounds the decoding-unit count INCLUSIVELY and then loops
 * to it inclusively as well, so the index reaches SEGMENTS:
 *
 *   ue(num_decoding_units_minus1, 0, HEVC_MAX_SLICE_SEGMENTS);   :1997
 *   for (i = 0; i <= current->num_decoding_units_minus1; i++) {  :2004
 *       ues(num_nalus_in_du_minus1[i], 0, HEVC_MAX_SLICE_SEGMENTS, 1, i);
 *
 * Index 600 of a uint16_t[600] sits at byte offset 1200, which is 4-aligned and
 * is exactly where du_cpb_removal_delay_increment_minus1 begins -- so the write
 * lands on that member's low half. The fix bounds the count by
 * FFMIN(pic_width_in_ctbs_y * pic_height_in_ctbs_y, HEVC_MAX_SLICE_SEGMENTS) - 1,
 * i.e. 599 at most.
 *
 * NOTHING WE HAVE CAN CATCH THIS, and that is the point of the row: the crossing
 * is inside one av_malloc, so a per-allocation bound is in bounds for it. The
 * oracle is therefore the upstream fix, not a protection. */
  struct pic_timing *pt = av_refstruct_allocz(sizeof *pt); /* cbs_sei.c:257 */
  CHECK(pt, 601);
  /* Offsets asserted rather than assumed: the whole claim is that index
   * SEGMENTS of the first member IS the first member of the second. */
  CHECK((char *)&pt->num_nalus_in_du_minus1[SEGMENTS]
        == (char *)&pt->du_cpb_removal_delay_increment_minus1[0], 602);
  CHECK(sizeof pt->num_nalus_in_du_minus1 == 1200, 603);

  const uint32_t sentinel = 0xA5A5A5A5u;
  pt->du_cpb_removal_delay_increment_minus1[0] = sentinel;

  /* Upstream's inclusive bound, and the fix's. */
  unsigned ndu = fixed ? (SEGMENTS - 1) : SEGMENTS;
  for (unsigned i = 0; i <= ndu; i++)
    pt->num_nalus_in_du_minus1[i] = 0x4141;

  uint32_t after = pt->du_cpb_removal_delay_increment_minus1[0];
  int clobbered = after != sentinel;
  /* The neighbour's LOW half is what index 600 overwrites; its high half must
   * survive, which is what makes this a sub-object crossing and not a wild
   * write. */
  int only_low_half = (after & 0xFFFF0000u) == (sentinel & 0xFFFF0000u);

  FF2_VERDICT(!fixed && clobbered && only_low_half,
              fixed && !clobbered,
              "index 600 of num_nalus_in_du_minus1[600] overwrote the low half of "
              "du_cpb_removal_delay_increment_minus1[0], inside one refstruct allocation",
              "the fix's bound of 599 keeps every write inside its own member");
  av_refstruct_unref(&pt);
  return !fixed ? !(clobbered && only_low_half) : !!clobbered;
}
