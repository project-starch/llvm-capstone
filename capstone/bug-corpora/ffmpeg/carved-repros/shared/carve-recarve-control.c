#include "corpus.h"

/* The re-carve control of the sublet-carve arm (FFC_SUBLET_CARVE only), run beside the cases in the
 * same boot. A 64-byte block, a 16-byte region carved from it and written once through its alias;
 * then, on the buggy arm, the region is carved again (ffc_recarve: its Sublet region is given back,
 * one revoke) and the OLD alias is written. That write must fault at the labelled probe: it is the
 * only evidence that the port's revocation exists, because no case in the corpus re-carves -- every
 * catch there comes from the bound the port sets on each region. The fixed arm writes the same byte
 * without the re-carve and must complete. */
#ifndef FFC_SUBLET_CARVE
#error "the re-carve control exists only for the sublet-carve arm"
#endif
FFC_CASE(98) {
  unsigned char *block = calloc(1, 64);
  CHECK(block, 981);
  unsigned char *region = ffc_carve(block, 0, 16, "control");
  write_probe_u8(region, 1); /* the live alias works */
  o->noted = 1;
  o->contained = 1;
  o->block = 64;
  o->region = 16;
  if (!fixed) {
    ffc_recarve(region);
    o->crossed = 1; /* reaching the next line means the stale alias was still usable */
  }
  write_probe_u8(region, 2);
  o->defect_text = "control: a region's old alias written after the region was carved again";
  o->fixed_text = "control: the same write, no re-carve";
  free(block);
}
