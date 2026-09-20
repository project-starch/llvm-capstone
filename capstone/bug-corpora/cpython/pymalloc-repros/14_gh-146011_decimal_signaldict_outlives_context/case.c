/* Case 14: gh-146011 -- use-after-free in signaldict_repr after deletion
 *
 * Shape: dangling pointer parked in a surviving object
 * Consumer: Modules/_decimal/_decimal.c
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

PYC_CASE(14) {
  /* gh-146011 -- _decimal's signal dict. traps->flags is a borrowed pointer
   * INTO the context object's own storage. context_clear released the
   * context while the signal dict, which can outlive it, kept the interior
   * pointer -- and signaldict_repr reads it whenever it is next called.
   *
   * The gap is unbounded here. Every other case's stale access happens
   * within the operation that created it, or at worst on the next API call;
   * this one waits for an unrelated repr() that may never come. */
  unsigned char *context = pym_malloc(OBJ);
  unsigned char *signaldict = pym_malloc(OBJ);
  CHECK(context && signaldict, 733);
  memset(context, 167, OBJ);
  signaldict[0] = 173;
  held = context + 16;       /* traps->flags, into the context's storage */
  pym_free(context);         /* context_clear, without clearing traps->flags */
  unsigned char *fresh = pym_malloc(OBJ);
  CHECK(fresh == context && signaldict[0] == 173, 734);
  memset(fresh, 179, OBJ);
  mark(14);
  (void)read_probe(held);    /* signaldict_repr, arbitrarily later */
}
