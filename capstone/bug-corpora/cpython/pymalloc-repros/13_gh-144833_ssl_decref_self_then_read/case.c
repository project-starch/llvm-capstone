/* Case 13: gh-144833 -- use-after-free in the SSL module when SSL_new() fails
 *
 * Shape: interior pointer into the object that was just released
 * Consumer: Modules/_ssl.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(13) {
  /* gh-144833 -- the SSL module when SSL_new() fails. The error path did
   * Py_DECREF(self) and then get_state_ctx(self), reading a field out of the
   * object it had just released.
   *
   * The stale access is to the freed object ITSELF, not to anything it
   * pointed at, and there is no second party involved at all. */
  unsigned char *self = pym_malloc(OBJ);
  CHECK(self, 731);
  memset(self, 157, OBJ);
  held = self + 8;           /* the ctx field inside self */
  pym_free(self);            /* Py_DECREF(self), first on the error path */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == self, 732);
  memset(fresh, 163, OBJ);
  mark(13);
  (void)read_probe(held);    /* get_state_ctx(self), second */
}
