#include "corpus.h"
#include <stdint.h>

/* The picture-private struct, reduced to the three reference-set arrays the
 * defect writes and the field that follows them. vulkan_hevc.c proves the
 * width is 8 in the pinned source itself, with memset(..., 0xff, 8) at :768;
 * the loop bound comes from a RefPicList that holds up to HEVC_MAX_REFS = 16. */
#define SET 8
struct hevc_pic_private {
  uint8_t RefPicSetStCurrBefore[SET];
  uint8_t RefPicSetStCurrAfter[SET];
  uint8_t RefPicSetLtCurr[SET];
  uint8_t tail[16];
};

FF2_CASE(1) {
/* Case 1 -- Vulkan HEVC reference sets, fix 68845e26f7. SUB-OBJECT, and the
 * most tightly contained row in the corpus: there is NO magnitude at which the
 * crossing leaves the allocation.
 *
 *   memset(hp->h265pic.RefPicSetStCurrBefore, 0xff, 8);                  :768
 *   for (int i = 0; i < h->rps[ST_CURR_BEF].nb_refs; i++)
 *       hp->h265pic.RefPicSetStCurrBefore[i] = j;                        :775
 *
 * The array holds 8 entries; nb_refs can reach HEVC_MAX_REFS = 16, so up to 8
 * bytes spill into the NEXT reference-set array of the same struct. The struct
 * is one refstruct allocation (decode.c:2352, frame_priv_data_size =
 * sizeof(HEVCVulkanDecodePicture)). The fix adds a three-way guard against
 * FF_ARRAY_ELEMS of each array.
 *
 * Vulkan-hwaccel builds only upstream; the reduction needs no Vulkan. */
  struct hevc_pic_private *hp = av_refstruct_allocz(sizeof *hp); /* decode.c:2352 */
  CHECK(hp, 611);
  CHECK((char *)&hp->RefPicSetStCurrBefore[SET] == (char *)&hp->RefPicSetStCurrAfter[0], 612);

  memset(hp->RefPicSetStCurrBefore, 0xff, SET); /* :768, upstream's own 8 */
  const uint8_t sentinel = 0x5a;
  memset(hp->RefPicSetStCurrAfter, sentinel, SET);

  /* nb_refs up to HEVC_MAX_REFS; the fix clamps it to FF_ARRAY_ELEMS. */
  int nb_refs = fixed ? SET : 16;
  for (int i = 0; i < nb_refs; i++)
    hp->RefPicSetStCurrBefore[i] = (uint8_t)i;

  int spilled = 0;
  for (int i = 0; i < SET; i++)
    if (hp->RefPicSetStCurrAfter[i] != sentinel)
      spilled++;
  /* Contained: the field AFTER the three arrays must be untouched, because the
   * spill is 8 bytes and lands wholly in the neighbouring array. */
  int contained = 1;
  for (int i = 0; i < 16; i++)
    if (hp->tail[i] != 0)
      contained = 0;

  FF2_VERDICT(!fixed && spilled == SET && contained,
              fixed && spilled == 0,
              "nb_refs = 16 wrote 8 bytes past RefPicSetStCurrBefore[8] into "
              "RefPicSetStCurrAfter, inside one refstruct allocation",
              "the fix's FF_ARRAY_ELEMS guard keeps every write inside its own array");
  av_refstruct_unref(&hp);
  return !fixed ? !(spilled == SET && contained) : !!spilled;
}
