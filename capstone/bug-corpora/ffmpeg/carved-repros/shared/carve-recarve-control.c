#include "corpus.h"

/* Both arms revoke and reissue an unaligned region at the same address,
 * checking that its successor and disjoint sibling still work. The buggy
 * arm then writes through the old reference, at the labelled probe. This
 * isolates temporal protection from the corpus's spatial-only crossings. */
#ifndef FFC_SUBLET_CARVE
#error "the re-carve control exists only for the Sublet carve"
#endif
FFC_CASE(98) {
  unsigned char *block = calloc(1, 64);
  CHECK(block, 981);
  unsigned char *region = ffc_carve(block, 3, 13, "control");
  unsigned char *sibling = ffc_carve(block, 19, 7, "sibling");
  write_probe_u8(region, 1); /* the live alias works */
  write_probe_u8(sibling, 3);
  ffc_recarve(region);
  unsigned char *replacement = ffc_carve(block, 3, 13, "successor");
  CHECK((unsigned long)(uintptr_t)replacement == (unsigned long)(uintptr_t)region, 982);
  write_probe_u8(replacement, 2);
  CHECK(read_probe_u8(sibling) == 3, 983);
  o->noted = 1;
  o->contained = 1;
  o->block = 64;
  o->region = 13;
  if (!fixed) {
    o->crossed = 1; /* reaching the next line means the stale alias was still usable */
  }
  write_probe_u8(fixed ? replacement : region, 2);
  o->defect_text = "control: a region's old alias written after the region was carved again";
  o->fixed_text = "control: the same write through the reissued region; sibling intact";
  free(block);
}
