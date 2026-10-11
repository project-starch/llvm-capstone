/* Case 11: gh-142783 -- possible use after free in the zoneinfo module
 *
 * Shape: free and use on adjacent lines
 * Consumer: Modules/_zoneinfo.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(11) {
  /* gh-142783 -- the zoneinfo weak cache. get_weak_cache asked for the
   * attribute, immediately Py_XDECREF'd it, and returned what the comment
   * called "a borrowed reference" on the assumption that the type held one.
   * When it does not, the object is gone before the caller's first use.
   *
   * No re-entrancy and no user callback: the free and the use are adjacent
   * lines. Every other case here needs something to run in between. */
  unsigned char *cache = pym_malloc(OBJ);
  CHECK(cache, 727);
  cache[0] = 137;
  held = cache;
  pym_free(cache);           /* Py_XDECREF, one line after the lookup */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == cache, 728);
  fresh[0] = 139;
  mark(11);
  (void)read_probe(held);    /* PyObject_CallMethod(weak_cache, "get", ...) */
}
