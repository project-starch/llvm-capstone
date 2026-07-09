// row11 control: csdrop consumes a HANDLE, it does not free memory.
//
// The distinction matters for the corpus claim. Row11 is not "the allocator
// refuses a second free"; it is "linearity leaves no second capability, so the
// second finalize has nothing to hand back". csdrop clears one register. It does
// not revoke a lineage, does not invalidate other capabilities over the same
// arena, and does not touch the bytes.
//
// So: carve the statement, drop it, and keep using the connection it came out
// of. If dropping the statement had swept anything, this would fault.
//
// It also gives the host independent evidence: the controller reads arena[8]
// through its ordinary Linux mmap and sees 0x44, the byte this domain wrote
// through the surviving low half AFTER the drop. Safe to read here because
// nothing revoked the arena's revocation node (see the guest controller's
// header comment for why that matters).
//
// Expected: no fault, retval 0x11130044, and arena[8] == 0x44 on the host.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *stmt = corpus_carve_stmt(); /* probe_arena is now the low half */
  stmt = __builtin_capstone_cap_drop(stmt); /* sqlite3_finalize(stmt) */
  (void)stmt;

  /* The connection the statement was carved from is untouched. */
  volatile char *conn = (volatile char *)probe_arena;
  conn[CORPUS_OFFSET] = 0x44;
  volatile char v = conn[CORPUS_OFFSET];

  *res = (CORPUS_RET_DROP_SIBLING_OK & ~0xffu) | (unsigned char)v; /* 0x11130044 */
}
