/* Case 17: gh-151416 -- borrowed ref use after free via fspath in os.spawnv/spawnve
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/posixmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(17) {
  /* gh-151416 -- os.spawnv/spawnve. The same __fspath__ trigger as case 16,
   * one module over, reached through a getitem function pointer rather than
   * a fast-sequence macro. Kept separate because it was reported and fixed
   * separately, months apart, which is the corpus's point about how narrowly
   * each of these gets patched. */
  void **argv = pym_malloc(OBJ);
  unsigned char *item = pym_malloc(OBJ);
  CHECK(argv && item, 739);
  item[0] = 211;
  argv[0] = item;            /* (*getitem)(argv, i), borrowed */
  pym_free(item);            /* __fspath__ mutated the list */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == item, 740);
  fresh[0] = 223;
  mark(17);
  (void)read_probe(argv[0]);
}
