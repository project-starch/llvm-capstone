/* Case 4: gh-145244 -- use-after-free on borrowed dict key in json encoder
 *
 * Shape: bulk free, stale read on the error path
 * Consumer: Modules/_json.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(4) {
  /* gh-145244 -- the JSON encoder's borrowed dict key, on the ERROR path.
   * key comes from PyDict_Next; the default callback clears the dict, which
   * frees every entry at once, and the error path then formats the key with
   * _PyErr_FormatNote("%R", key). The stale access happens while unwinding,
   * which is where a fault is least expected.
   *
   * Verified live at the pin by reading the source, not by the apply test:
   * Modules/_json.c:1621 of v3.13.7 passes key and value to
   * encoder_encode_key_value with no Py_INCREF at all. */
  unsigned char *entries[4];
  for (unsigned i = 0; i < 4; ++i) {
    entries[i] = pym_malloc(OBJ);
    CHECK(entries[i], 709);
    entries[i][0] = (unsigned char)(50 + i);
  }
  held = entries[2];         /* the key the error path will format */
  for (unsigned i = 0; i < 4; ++i)
    pym_free(entries[i]);    /* PyDict_Clear, in the callback */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh, 710);
  fresh[0] = 59;
  mark(4);
  (void)read_probe(held);
}
