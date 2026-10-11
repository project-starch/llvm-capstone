/* Case 1: gh-146613 -- re-entrant use-after-free in itertools._grouper
 *
 * Shape: free / reuse / stale read, with a surviving sibling alias
 * Consumer: Modules/itertoolsmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(1) {
  /* gh-146613 -- itertools._grouper, the child iterator. Two keys are live
   * at once: igo->tgtkey in the grouper and gbo->currkey in the parent. The
   * comparison frees the parent's while the grouper is still holding its
   * borrowed alias. */
  unsigned char *tgtkey = pym_malloc(OBJ);
  unsigned char *currkey = pym_malloc(OBJ);
  CHECK(tgtkey && currkey && tgtkey != currkey, 703);
  tgtkey[0] = 23;
  currkey[0] = 29;
  held = currkey;
  pym_free(currkey);                          /* the parent advanced */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == currkey && tgtkey[0] == 23, 704);
  fresh[0] = 31;
  mark(1);
  (void)read_probe(held);
}
