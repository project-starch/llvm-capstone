/* Case 15: gh-149449 -- use-after-free in _PyUnicode_GetNameCAPI
 *
 * Shape: bare PyMem block, cached by a third party
 * Consumer: Modules/unicodedata.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(15) {
  /* gh-149449 -- unicodedata's capsule. _PyUnicode_Name_CAPI was a raw
   * PyMem_Malloc block owned by a capsule; other code cached the pointer,
   * and when unicodedata left sys.modules the capsule's destructor freed it
   * under them. Upstream's fix was to make the struct static.
   *
   * The freed thing is not a PyObject at all -- it is a bare allocation, and
   * it is pymalloc's because obmalloc sets PYMEM_DOMAIN_MEM to
   * PYMALLOC_ALLOC, so PyMem_Malloc reaches the same pools that
   * PyObject_Malloc does. That is the fact this case exists to exercise. */
  unsigned char *capi = pym_malloc(32); /* the _PyUnicode_Name_CAPI block */
  CHECK(capi, 735);
  memset(capi, 191, 32);
  held = capi;
  pym_free(capi);            /* the capsule's destructor, at module teardown */
  unsigned char *fresh = pym_malloc(32);
  CHECK(fresh == capi, 736);
  fresh[0] = 193;
  mark(15);
  (void)read_probe(held);    /* the cached capi->getname, called later */
}
