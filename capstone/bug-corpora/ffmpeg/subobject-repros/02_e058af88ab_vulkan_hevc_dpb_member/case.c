#include "corpus.h"
#include <stdint.h>

/* The picture-private struct, reduced to the two members the DPB walk fills.
 * vulkan_hevc.c:123-125 declares them HEVC_MAX_REFS wide; the walk is driven by
 * l->DPB, which hevcdec.h:453 declares as HEVCFrame DPB[32]. */
#define REFS 16
#define DPB 32
struct hevc_dpb_private {
  void *ref_src[REFS];
  uint32_t h265_refs[REFS];
};

FF2_CASE(2) {
/* Case 2 -- Vulkan HEVC DPB fill, fix e058af88ab. SUB-OBJECT: the first
 * out-of-bounds write lands on the immediately following member.
 *
 *   for (int i = 0; i < FF_ARRAY_ELEMS(l->DPB); i++) {
 *       ...
 *       int idx = nb_refs;
 *       err = vk_hevc_fill_pict(avctx, &hp->ref_src[idx], ...);           :760
 *
 * The walk runs over DPB[32] while the targets are HEVC_MAX_REFS = 16 wide, so
 * idx can reach 31 and the FIRST out-of-bounds write is ref_src[16], which is
 * h265_refs[0] -- the next member of the same refstruct allocation
 * (decode.c:2352). The fix adds "if (nb_refs >= HEVC_MAX_REFS) return
 * AVERROR_INVALIDDATA;".
 *
 * Reduced to the first crossing write. Far past it the walk would leave the
 * struct too; the bound crossed FIRST is the sub-object one, and that is what
 * this row claims. */
  struct hevc_dpb_private *hp = av_refstruct_allocz(sizeof *hp); /* decode.c:2352 */
  CHECK(hp, 621);
  CHECK((char *)&hp->ref_src[REFS] == (char *)&hp->h265_refs[0], 622);

  const uint32_t sentinel = 0xA5A5A5A5u;
  for (int i = 0; i < REFS; i++)
    hp->h265_refs[i] = sentinel;

  /* nb_refs advances once per usable DPB entry; the fix stops it at REFS. The
   * buggy arm is reduced to ONE write past the array -- the FIRST crossing,
   * ref_src[16]. Writing all 16 extra entries the DPB[32] walk permits would
   * run 128 bytes into a 64-byte member and leave the allocation entirely,
   * which is a different claim; the first attempt at this case did that and the
   * containment check correctly refused it. */
  int writes = fixed ? REFS : REFS + 1;
  int nb_refs = 0;
  for (int i = 0; i < DPB && nb_refs < writes; i++)
    hp->ref_src[nb_refs++] = (void *)(uintptr_t)(0x1000 + i);

  /* The first crossing overwrites h265_refs[0] (and [1], a pointer being 8
   * bytes and the member 4). */
  int clobbered = hp->h265_refs[0] != sentinel;
  /* Contained: the far end of the second member must survive, so the write is
   * a member-to-member crossing and not a run off the allocation. */
  int contained = hp->h265_refs[REFS - 1] == sentinel;

  FF2_VERDICT(!fixed && clobbered && contained,
              fixed && !clobbered,
              "the DPB walk wrote ref_src[16], which is h265_refs[0], inside one "
              "refstruct allocation",
              "the fix's nb_refs >= HEVC_MAX_REFS guard stops the walk at the array's end");
  av_refstruct_unref(&hp);
  return !fixed ? !(clobbered && contained) : !!clobbered;
}
