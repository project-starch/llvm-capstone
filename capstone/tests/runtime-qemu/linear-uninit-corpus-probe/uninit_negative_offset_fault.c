// row14, the sharp edge: UNINIT denies reads by TYPE, not by cursor placement.
//
// csrevoke parks the retained handle's cursor at `end`, so an ordinary forward
// read -- `*db`, or `db[k]` for k >= 0 -- is already out of bounds. A probe that
// only tested those would pass on an emulator that never looks at the capability
// type at all, and would be evidence for the wrong claim: "reads past the end of
// an uninitialised region trap", rather than row14's "an uninitialised
// capability has no read authority".
//
// A NEGATIVE offset separates the two. `db[-1]` addresses end-1, which is inside
// [base, end): it clears the bounds check and, before the type check existed,
// returned the stale byte -- the exact disclosure UNINIT is there to prevent.
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
  volatile char *p = (volatile char *)db; /* cursor == end */

  volatile char v = p[-1]; /* IN BOUNDS, still no read authority -> FAULT */

  *res = CORPUS_RET_UNINIT_NEG_NOTRAP | (unsigned char)v; /* unreachable */
}
