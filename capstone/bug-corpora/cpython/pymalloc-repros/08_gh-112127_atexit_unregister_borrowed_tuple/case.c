/* Case 8: gh-112127 -- possible use-after-free in atexit.unregister()
 *
 * Shape: free / reuse / stale read
 * Consumer: Modules/atexitmodule.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(8) {
  /* gh-112127 -- atexit.unregister(). The loop compares the caller's func
   * against each registered callback, borrowing the tuple out of the live
   * callbacks list. PyObject_RichCompareBool runs a user __eq__, which can
   * call atexit.unregister again and mutate the list, dropping the tuple the
   * comparison is standing on -- and the loop then carries on to the next
   * index through the same list. */
  void **callbacks = pym_malloc(OBJ);   /* the list's ob_item */
  unsigned char *tuple = pym_malloc(OBJ);
  unsigned char *other = pym_malloc(OBJ);
  CHECK(callbacks && tuple && other, 719);
  tuple[0] = 97;
  other[0] = 101;
  callbacks[0] = tuple;
  callbacks[1] = other;
  pym_free(tuple);           /* the re-entrant unregister removed it */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == tuple && other[0] == 101, 720);
  fresh[0] = 103;
  mark(8);
  (void)read_probe(callbacks[0]);
}
