#include "corpus.h"

/* The carve control, built and run beside the cases on every CheriBSD arm. A 64-byte block, a
 * 16-byte region carved at its start, and a one-byte write at region + 16: inside the block,
 * outside the region. Under FFC_CARVE_BOUNDS it must die by SIGPROT -- the narrowing is shown to
 * work in this boot -- and on every other arm it must complete, showing in the same boot that the
 * platform's own allocation bound does not see a crossing that stays inside the block. */
FFC_CASE(99) {
  unsigned char *block = calloc(1, 64);
  CHECK(block, 991);
  unsigned char *region = ffc_carve(block, 0, 16, "control");
  if (fixed) {
    ffc_note(o, block, 64, region, 16, region + 15, 1);
    write_probe_u8(region + 15, 1);
  } else {
    ffc_note(o, block, 64, region, 16, region + 16, 1);
    write_probe_u8(region + 16, 1);
  }
  o->defect_text = "control: one byte past a 16-byte carve, inside its 64-byte block";
  o->fixed_text = "control: the carve's last byte";
  free(block);
}
