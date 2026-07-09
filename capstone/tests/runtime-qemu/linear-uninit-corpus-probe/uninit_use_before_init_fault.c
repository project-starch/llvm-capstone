// row14 cpython_uninit_connection -- UNINIT, use-before-init. THE MECHANISM.
//
// Stage-1 defect: a calloc'd connection wrapper is handed out with its
// `sqlite3 *db` still NULL, and `*(unsigned char *)connection->db` segfaults.
//
// Stage-2 shape: `db` is an UNINITIALISED capability. It names the connection's
// storage but carries no read authority; only a successful sqlite3_open --
// csinit -- turns it into a usable handle. A read before that must fault, and
// must fault BECAUSE the capability is uninitialised, not because some address
// happened to be out of range.
//
// The derivation is intra-domain and needs no monitor op: revoke a lineage that
// is still LINEAR and the retained handle comes back UNINIT (corpus_domain.h).
//
// Expected: FAULT, cause 26 (RISCV_EXCP_UNEXP_CAP_TYPE, "Cap mem load through
// uninitialised capability"). Cause 26 is self-proving -- only an UNINIT-typed
// capability raises it, so unlike a cause-24 (tag gone) expectation this needs
// no companion control to rule out an unrelated cause. uninit_init_then_use_ok
// is still run, to show the memory itself is fine and csinit reclaims it.
#include "corpus_domain.h"

void domain_main(void *arg, unsigned func) {
  if (probe_receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  void *db = corpus_uninit_handle(); /* UNINIT: names the bytes, cannot read */
  volatile char *p = (volatile char *)db;

  volatile char v = *p; /* USE BEFORE INIT -> must FAULT */

  *res = CORPUS_RET_UNINIT_NOTRAP | (unsigned char)v; /* unreachable */
}
