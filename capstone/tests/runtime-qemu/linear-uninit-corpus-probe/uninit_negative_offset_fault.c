// row14, the sharp edge: UNINIT denies reads by TYPE, not by cursor placement.
//
// The claim to separate is "an uninitialised capability has no READ AUTHORITY" from the weaker
// "reads outside the region trap". A probe that only read out-of-bounds addresses would pass on an
// emulator that never looks at the capability type at all, and would be evidence for the wrong one.
//
// REWRITTEN 2026-09-10, because its own mechanism was inverted by R-31. Revoke used to park the
// cursor at `end`, so this probe read `db[-1]` -- which was then `end-1`, INSIDE [base, end) -- and
// that was the whole point. Revoke now returns the cursor at BASE, so `db[-1]` addresses `base-1`,
// which is OUTSIDE. It still reported the expected cause, because the type check runs before the
// bounds check, so it kept PASSING while testing the opposite of what it says.
//
// A well-inside offset restores the discrimination under the new cursor placement: CORPUS_OFFSET is
// comfortably within the region and is not where the cursor sits, so a pass still means the refusal
// came from the TYPE and not from bounds.
//
// So this probe fails (returns instead of faulting) on an emulator without the
// uninit-load check in _helper_access_with_cap, and it is the reason that check
// was added. It also pins the cause apart from the bounds faults: 26, not 5.
//
// Expected: FAULT, cause 26 (RISCV_EXCP_UNEXP_CAP_TYPE).
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *db = corpus_uninit_handle();
  volatile char *p = (volatile char *)db; /* cursor == BASE since R-31 */

  volatile char v = p[CORPUS_OFFSET]; /* IN BOUNDS, still no read authority -> FAULT */

  *res = CORPUS_RET_UNINIT_NEG_NOTRAP | (unsigned char)v; /* unreachable */
}
