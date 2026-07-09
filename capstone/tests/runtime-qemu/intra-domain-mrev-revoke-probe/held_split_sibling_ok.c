// Can the domain revoke PART of its granted arena and keep using the rest?
//
// This is the property the SQLite linear-heap direction (Option B step 3) rests
// on: memsys5 hands out sub-buffers of one arena, so revoking a freed allocation
// must not invalidate its neighbours. SPLIT gives each half a fresh
// revocation-tree node at the same depth; SHRINK/SHRINKTO copy rev_node_id
// unchanged and are NOT a substitute (task-005, finding Q3). task-005 proved this
// on a csdebuggencap-minted arena; this re-proves it on a real monitor grant.
//
//   arena = <monitor grant>            [base, base+4096)
//   hi    = split(arena, base+2048)    arena := [base, base+2048), hi := [base+2048, ...)
//   R     = mrev(hi); revoke(R)        sweeps the HIGH half's lineage only
//   deref arena                        the low sibling must survive
//
// Expected: no fault, retval 0x22350044.
#include "probe_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *A = probe_arena; /* LIN [base, base + PROBE_REGION_SIZE) */
  unsigned long base = __builtin_capstone_cap_get_base(A);

  void *hi = probe_cap_split(&A, base + PROBE_SPLIT_MID); /* A keeps the low half */
  void *R = __builtin_capstone_cap_mrev(hi); /* rev handle over the high half */

  volatile char *lo = (volatile char *)A;
  lo[PROBE_OFFSET] = 0x44; /* low half, live */

  __builtin_capstone_cap_revoke(R); /* high half only */

  volatile char v = lo[PROBE_OFFSET]; /* sibling untouched by the sweep */

  *res = (PROBE_RET_SPLIT_OK & ~0xffu) | (unsigned char)v; /* 0x22350044 */
}
