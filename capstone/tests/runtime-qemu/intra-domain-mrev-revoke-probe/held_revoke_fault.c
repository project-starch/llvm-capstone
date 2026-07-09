// PRIMARY probe: the literal single-domain Option B "gold standard".
//
// One domain receives a real monitor-granted LINEAR arena, mints its own
// revocation handle over it, hands itself a delinearised working alias, uses the
// alias while the grant is live, REVOKEs at a lifecycle point, and then touches
// the cached alias again. That last touch must fault.
//
//   arena = <monitor REGION_SHARE grant, LIN, RW>
//   R     = mrev(arena)     // retains arena (task-005, finding Q2)
//   alias = delin(arena)    // copyable, still revocable (task-005, finding C3)
//   *alias = sentinel       // live use through the held capability: ok
//   revoke(R)               // lifecycle point
//   read *alias             // USE-AFTER-REVOKE -> must FAULT
//
// Everything happens inside ONE domain entry so the alias stays in a register
// across the revoke; that is what makes the fault self-proving. Cause 25 ("Cap
// mem access on revoked capability") means the tag was INTACT and the revoked
// rev-tree node stopped the deref. Cause 24 would mean the tag was gone, which a
// consumed linear capability also produces -- see held_mem_alias_fault.c, whose
// alias is deliberately memory-resident, and its no-revoke control.
#include "probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg; /* CALL entry: arg is the retval slot */

  void *A = probe_arena;                     /* LIN, monitor-granted */
  void *R = __builtin_capstone_cap_mrev(A);  /* owner keeps the rev handle */
  void *D = __builtin_capstone_cap_delin(A); /* copyable working alias */
  volatile char *buf = (volatile char *)D;

  buf[PROBE_OFFSET] = (char)PROBE_SENTINEL_LIVE; /* live use: ok */
  __builtin_capstone_cap_revoke(R);              /* lifecycle point */
  volatile char v = buf[PROBE_OFFSET];           /* USE-AFTER-REVOKE -> FAULT */

  *res = PROBE_RET_REVOKE_NOTRAP | (unsigned char)v; /* unreached if it faults */
}
