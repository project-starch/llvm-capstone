/* Case 6: gh-151295 -- use-after-free in bytes.join()/bytearray.join() via re-entrant __buffer__
 *
 * Shape: payload buffer, sub-512
 * Consumer: Objects/stringlib/join.h
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(6) {
  /* gh-151295 -- bytes.join()/bytearray.join() through a re-entrant
   * __buffer__. buffers[i].buf points INTO the item's payload; the item's
   * __buffer__ runs Python that drops the sequence's last reference to it.
   * The join then copies from the released buffer.
   *
   * SIZE MATTERS HERE AND NOWHERE ELSE IN THIS CORPUS. The payload is pinned
   * below pymalloc's 512-byte threshold, which is what makes the defect
   * invisible to a malloc-level tool. On a large input the same defect is an
   * ordinary malloc use-after-free and ASan reports it. */
  unsigned char *payload = pym_malloc(OBJ);
  CHECK(payload, 714);
  memset(payload, 73, OBJ);
  held = payload + 8;        /* buffers[i].buf, an interior pointer */
  pym_free(payload);         /* __buffer__ dropped the sequence's last ref */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == payload, 715);
  memset(fresh, 79, OBJ);
  mark(6);
  (void)read_probe(held);    /* the join's memcpy source */
}
