#include "../shared/corpus.h"

/* CONTROL, not a case: one pymalloc block, freed, then read through the alias
 * that outlived it -- the labelled read_probe, with nothing reallocated in
 * between. It says what the arm IS before any case is scored: pymalloc stock
 * keeps the freed block inside its pool, so the read COMPLETES; the lifetime
 * adapter issues each block as its own capability and retires it on free, so
 * the read FAULTS in read_probe. tools/arms.json records both. */
PYC_CASE(0) {
  unsigned char *block = pym_malloc(OBJ);
  CHECK(block != NULL, 791);
  block[0] = 67;
  held = block;
  pym_free(block);
  mark(0);
  (void)read_probe(held);
}
