#include "corpus.h"
#include <stdint.h>

/* AV1VulkanDecodePicture's shape around tile_sizes, as vulkan_av1.c:37-50
 * declares it at the pin. MAX_TILES is 256 (vulkan_av1.c:24). The struct is ONE
 * allocation -- it is the hwaccel_picture_private of the current frame -- so the
 * bound between tile_sizes and std_ref is not a bound any allocator knows about.
 *
 * std_ref is a StdVideoDecodeAV1ReferenceInfo, a Vulkan header type that is not
 * available here; it is reduced to a byte array of a plausible size, because
 * nothing about this defect depends on its contents -- only on its being the
 * member that follows tile_sizes. */
#define MAX_TILES 256
#define STD_REF_BYTES 32

struct av1_vulkan_decode_picture {
  uint32_t tile_sizes[MAX_TILES];
  uint8_t std_ref[STD_REF_BYTES]; /* StdVideoDecodeAV1ReferenceInfo */
};

/* The crossing, labelled so a fault can be required to land HERE. */
__attribute__((noinline, used)) static void
write_probe(volatile uint32_t *p, uint32_t v) {
  *p = v;
}

FF2_CASE(5) {
  /* Case 5 -- Vulkan AV1 slice decode, fix d29ff88422. SUB-OBJECT: a four-byte
   * WRITE from the end of one struct member into the next, inside a single
   * allocation.
   *
   * At the pin, vulkan_av1.c:573-578 bounds the tile count ONCE, before the
   * loop, and with the wrong relation:
   *
   *     if (ap->av1_pic_info.tileCount > MAX_TILES)
   *         return AVERROR(ENOSYS);
   *     for (int i = s->tg_start; i <= s->tg_end; i++) {
   *         ap->tile_sizes[ap->av1_pic_info.tileCount] = ...;
   *
   * tileCount is then incremented inside the loop (by ff_vk_decode_add_slice's
   * caller), so a tile group that starts at exactly MAX_TILES - 1 writes index
   * MAX_TILES on its second iteration, and every iteration after that writes
   * further out. Two separate errors compound: the check is `>` where it must be
   * `>=`, and it is hoisted out of the loop it is supposed to guard. The fix
   * moves it INSIDE the loop and makes it `>=`, which is one logical change
   * expressed as two lines moving.
   *
   * Index MAX_TILES of a uint32_t[256] is byte offset 1024, which is exactly
   * where std_ref begins -- so the write lands on that member's first four
   * bytes. NOTHING WE HAVE CAN CATCH THIS: the crossing is inside one
   * allocation, so a per-allocation bound is in bounds for it. */
  struct av1_vulkan_decode_picture *ap = av_refstruct_allocz(sizeof *ap);
  CHECK(ap, 631);
  /* The whole claim: index MAX_TILES of the first member IS the second member. */
  CHECK((char *)&ap->tile_sizes[MAX_TILES] == (char *)&ap->std_ref[0], 632);
  CHECK(sizeof ap->tile_sizes == 1024, 633);

  const uint32_t sentinel = 0xA5A5A5A5u;
  memcpy(ap->std_ref, &sentinel, sizeof sentinel);

  /* The loop, with the count arriving one short of the limit -- the reachable
   * case the fix's `>=` rejects and the pin's `>` admits. */
  unsigned tile_count = MAX_TILES - 1;
  const unsigned iterations = 2; /* tg_start..tg_end inclusive, two tile groups */
  unsigned wrote_out_of_member = 0;

  for (unsigned i = 0; i < iterations; i++) {
    if (fixed) {
      /* The fix: the check is inside the loop, and it is >=. */
      if (tile_count >= MAX_TILES)
        break;
    }
    if (tile_count >= MAX_TILES)
      wrote_out_of_member++;
    write_probe(&ap->tile_sizes[tile_count], 0x41414141u);
    tile_count++;
  }

  uint32_t after;
  memcpy(&after, ap->std_ref, sizeof after);
  int clobbered = after != sentinel;
  /* Only the neighbour's FIRST four bytes may move; the rest must survive, which
   * is what makes this a sub-object crossing and not a wild write. */
  int rest_survived = 1;
  for (unsigned i = sizeof(uint32_t); i < STD_REF_BYTES; i++)
    if (ap->std_ref[i] != 0)
      rest_survived = 0;

  printf("cap=%zu touched=%u out_of_member=%u clobbered=%d\n",
         sizeof ap->tile_sizes, tile_count, wrote_out_of_member, clobbered);

  FF2_VERDICT(!fixed && clobbered && wrote_out_of_member == 1 && rest_survived,
              fixed && !clobbered && wrote_out_of_member == 0,
              "the pre-loop `> MAX_TILES` check let tileCount reach 256 and the write "
              "landed on std_ref's first four bytes, inside one allocation",
              "the fix's in-loop `>= MAX_TILES` check stops before index 256 is formed");
  av_refstruct_unref(&ap);
  return !fixed ? !(clobbered && rest_survived) : !!clobbered;
}
