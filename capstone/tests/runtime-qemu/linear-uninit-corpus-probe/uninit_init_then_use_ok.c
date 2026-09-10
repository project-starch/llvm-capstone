// row14 control: after sqlite3_open succeeds, the connection works.
//
// The other half of the row's claim. uninit_use_before_init_fault shows the
// pre-init read traps; this shows the trap was about the capability's TYPE and
// not about broken memory, a dead region or a botched derivation: the very same
// handle, put through csinit, reads and writes the very same bytes.
//
//   db     = <UNINIT handle over the arena>     no read authority, cursor == BASE
//   opened = fill(db) then init(db, 0)          sqlite3_open(): LIN, cursor = base
//   d      = delin(opened)                      an ordinary pointer to hand out
//   d[8]   = 0x5e; read it back                 the connection is usable
//
// csinit is a required, explicit reclaim step: it asserts the input is UNINIT with the cursor
// having REACHED end, and it consumes it (UNINIT is not copyable). There is no way to skip it and
// no second handle left behind -- and since R-31 no way to reach it without first overwriting the
// borrower's data, which is the point.
//
// Expected: no fault, retval 0x1412005e.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *db = corpus_uninit_handle();
  /* FILL, then init. Was a bare cap_init: revoke used to hand the cursor back AT end, so csinit
   * succeeded with no rewrite of the borrower's data at all -- the disclosure R-31 closes. Now the
   * cursor comes back at BASE and csinit refuses it until the region has been overwritten. */
  void *opened = corpus_fill_then_init(db, CORPUS_REGION_SIZE); /* LIN over [base, end) */
  void *d = __builtin_capstone_cap_delin(opened);

  volatile char *p = (volatile char *)d;
  p[CORPUS_OFFSET] = (char)CORPUS_SENTINEL;
  volatile char v = p[CORPUS_OFFSET];

  *res = (CORPUS_RET_INIT_OK & ~0xffu) | (unsigned char)v; /* 0x1412005e */
}
