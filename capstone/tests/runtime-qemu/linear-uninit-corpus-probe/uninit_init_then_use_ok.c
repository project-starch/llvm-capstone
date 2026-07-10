// row14 control: after sqlite3_open succeeds, the connection works.
//
// The other half of the row's claim. uninit_use_before_init_fault shows the
// pre-init read traps; this shows the trap was about the capability's TYPE and
// not about broken memory, a dead region or a botched derivation: the very same
// handle, put through csinit, reads and writes the very same bytes.
//
//   db     = <UNINIT handle over the arena>     no read authority, cursor == end
//   opened = init(db, 0)                        sqlite3_open(): LIN, cursor = base
//   d      = delin(opened)                      an ordinary pointer to hand out
//   d[8]   = 0x5e; read it back                 the connection is usable
//
// csinit is a required, explicit reclaim step: it asserts the input is UNINIT
// with cursor == end, and it consumes it (UNINIT is not copyable). There is no
// way to skip it and no second handle left behind.
//
// Expected: no fault, retval 0x1412005e.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *db = corpus_uninit_handle();
  void *opened = __builtin_capstone_cap_init(db, 0); /* LIN over [base, end) */
  void *d = __builtin_capstone_cap_delin(opened);

  volatile char *p = (volatile char *)d;
  p[CORPUS_OFFSET] = (char)CORPUS_SENTINEL;
  volatile char v = p[CORPUS_OFFSET];

  *res = (CORPUS_RET_INIT_OK & ~0xffu) | (unsigned char)v; /* 0x1412005e */
}
