/* Case 0: gh-143543 -- re-entrant use-after-free in itertools.groupby
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/itertoolsmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(0) {
  /* gh-143543 -- itertools.groupby. groupby compares gbo->tgtkey with
   * gbo->currkey, both borrowed. A user-defined __eq__ re-enters the
   * iterator and advances it, dropping the last reference to the key being
   * compared; the comparison then continues through the stale pointer. */
  unsigned char *key = pym_malloc(OBJ);
  CHECK(key, 701);
  key[0] = 17;
  held = key;                /* gbo->currkey, borrowed by the comparison */
  pym_free(key);             /* the re-entrant __eq__ advanced the iterator */
  unsigned char *successor = pym_malloc(OBJ); /* the next group's key */
  CHECK(successor == key, 702); /* the block comes straight back */
  successor[0] = 19;
  mark(0);
  (void)read_probe(held);
}
