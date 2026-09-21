/* Case 12: gh-143004 -- possible use-after-free in collections.Counter.update()
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/_collectionsmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(12) {
  /* gh-143004 -- collections.Counter.update via _count_elements. oldval is
   * borrowed from the mapping; PyNumber_Add runs a user __add__ that can
   * mutate or clear the dict, freeing the value while the sum is being
   * computed. The mapping itself survives, which is what separates this from
   * case 4: there the container was emptied and abandoned, here it is
   * emptied and kept. */
  void **values = pym_malloc(OBJ);
  unsigned char *oldval = pym_malloc(OBJ);
  CHECK(values && oldval, 729);
  oldval[0] = 149;
  values[0] = oldval;        /* the dict's slot, borrowed as oldval */
  pym_free(oldval);          /* the user __add__ cleared the dict */
  values[0] = NULL;          /* ... and the container is still live */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == oldval, 730);
  fresh[0] = 151;
  held = oldval;             /* PyNumber_Add is still holding it */
  mark(12);
  (void)read_probe(held);
}
