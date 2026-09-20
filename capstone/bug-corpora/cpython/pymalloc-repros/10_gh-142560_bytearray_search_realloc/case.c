/* Case 10: gh-142560 -- use-after-free in bytearray search-like methods
 *
 * Shape: realloc moved the block
 * Consumer: Objects/bytearrayobject.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(10) {
  /* gh-142560 -- bytearray's search-like methods. They cache
   * PyByteArray_AS_STRING(self) and then call into code that can run user
   * Python, which may resize the bytearray. The resize REALLOCATES the
   * storage, and when it moves, the cached base pointer is left addressing
   * the old block.
   *
   * This is the only case in the corpus where the block is ended by a
   * REALLOC rather than a free, so the assertion that it really moved is
   * part of the case: a realloc that returned the same address would leave
   * the driver testing nothing at all. */
  unsigned char *storage = pym_malloc(OBJ);
  CHECK(storage, 723);
  memset(storage, 113, OBJ);
  held = storage;            /* the cached PyByteArray_AS_STRING(self) */
  unsigned char *grown = pym_realloc(storage, 300); /* user code resized it */
  CHECK(grown, 724);
  CHECK(grown != storage, 725); /* it MUST have moved, or this tests nothing */
  grown[0] = 127;
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == storage, 726); /* the old block came back */
  memset(fresh, 131, OBJ);
  mark(10);
  (void)read_probe(held);    /* _Py_bytes_find over the old base */
}
