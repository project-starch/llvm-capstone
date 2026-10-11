/* Case 2: gh-142829 -- use-after-free in Context.__eq__ via re-entrant ContextVar.set
 *
 * Shape: interior pointer into a freed block
 * Consumer: Python/hamt.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(2) {
  /* gh-142829 -- Context.__eq__ through _PyHamt_Eq. The comparison walks the
   * map with an iterator whose state points INSIDE the node it is visiting;
   * a re-entrant ContextVar.set drops the last reference to that node, and
   * the walk resumes from the interior pointer. */
  unsigned char *hamt_node = pym_malloc(OBJ);
  CHECK(hamt_node, 705);
  memset(hamt_node, 37, OBJ);
  held = hamt_node + 16;     /* iter.i_nodes[level], mid-node */
  pym_free(hamt_node);       /* the re-entrant set dropped the last ref */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == hamt_node, 706);
  memset(fresh, 41, OBJ);
  mark(2);
  (void)read_probe(held);
}
