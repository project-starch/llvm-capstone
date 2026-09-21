/* Case 3: gh-142831 -- use-after-free in json encoder during re-entrant mutation
 *
 * Shape: stale entry reached through a live array
 * Consumer: Modules/_json.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(3) {
  /* gh-142831 -- the JSON encoder over a dict's items list. The item is
   * borrowed from a list that stays live; user code invoked from the encoder
   * mutates the list, dropping the item. The stale pointer is therefore
   * reached through a LIVE array, not through a local. */
  void **ob_item = pym_malloc(OBJ); /* the items list's storage */
  unsigned char *item = pym_malloc(OBJ);
  CHECK(ob_item && item, 707);
  item[0] = 43;
  ob_item[0] = item;         /* PyList_GET_ITEM(items, i), borrowed */
  pym_free(item);            /* the default callback mutated the list */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == item, 708);
  fresh[0] = 47;
  mark(3);
  (void)read_probe(ob_item[0]);
}
