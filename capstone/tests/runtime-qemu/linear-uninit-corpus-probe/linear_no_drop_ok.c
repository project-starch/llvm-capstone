// row11 control: without the drop, the statement handle works.
//
// Required, because both row11 fault probes assert cause 24 ("Cap mem access
// requires capability" / "DROP requires capability"), and cause 24 means only
// that the register held no capability. A carve that silently produced a null
// handle, or a linear capability consumed by an earlier copy (task-005, finding
// C3), would fault identically. This probe runs the identical carve and deref
// with the csdrop removed, and must reach its return.
//
// Expected: no fault, retval 0x11120033.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *stmt = corpus_carve_stmt(); /* LINEAR, cursor at its own base */
  volatile char *p = (volatile char *)stmt;

  p[CORPUS_OFFSET] = 0x33;                /* the statement is live... */
  volatile char v = p[CORPUS_OFFSET];      /* ...and readable */

  *res = (CORPUS_RET_NO_DROP_OK & ~0xffu) | (unsigned char)v; /* 0x11120033 */
}
